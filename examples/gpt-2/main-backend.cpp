#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "common.h"
#include "common-ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define GPT2_MAX_NODES 4096

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

// default hparams (GPT-2 117M)
struct gpt2_hparams {
    int32_t n_vocab = 50257;
    int32_t n_ctx   = 1024;
    int32_t n_embd  = 768;
    int32_t n_head  = 12;
    int32_t n_layer = 12;
    int32_t ftype   = 1;
    float   eps     = 1e-5f;
};

struct gpt2_layer {
    // normalization
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    struct ggml_tensor * ln_2_g;
    struct ggml_tensor * ln_2_b;

    // attention
    struct ggml_tensor * c_attn_attn_w;
    struct ggml_tensor * c_attn_attn_b;

    struct ggml_tensor * c_attn_proj_w;
    struct ggml_tensor * c_attn_proj_b;

    // mlp
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w;
    struct ggml_tensor * c_mlp_proj_b;
};

struct gpt2_model {
    gpt2_hparams hparams;

    // normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    struct ggml_tensor * wte;     //    token embedding
    struct ggml_tensor * wpe;     // position embedding
    struct ggml_tensor * lm_head; // language model head

    std::vector<gpt2_layer> layers;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    //
    struct ggml_context * ctx_w;
    struct ggml_context * ctx_kv;

    ggml_backend_t backend = NULL;

    ggml_backend_buffer_t buffer_w;
    ggml_backend_buffer_t buffer_kv;

    std::map<std::string, struct ggml_tensor *> tensors;
};

// load the model's weights from a file
bool gpt2_model_load(const std::string & fname, gpt2_model & model, gpt_vocab & vocab, int n_ctx, int n_gpu_layers) {
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.ftype,   sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: ftype   = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr   = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        fin.read((char *) &n_vocab, sizeof(n_vocab));

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        std::vector<char> buf(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            buf.resize(len);
            fin.read((char *) buf.data(), len);
            word.assign(buf.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }

    ggml_log_set(ggml_log_callback_default, nullptr);

    auto & ctx = model.ctx_w;

    // create the ggml context
    {
        size_t n_tensors = 2 + 6 + 12*model.hparams.n_layer;
        struct ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        ctx = ggml_init(params);
        if (!ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // initialize the backend
#ifdef GGML_USE_CUDA
    if (n_gpu_layers > 0) {
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0);
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (n_gpu_layers > 0) {
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        model.backend = ggml_backend_metal_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    }
#endif

    if (!model.backend) {
        // fallback to CPU backend
        fprintf(stderr, "%s: using CPU backend\n", __func__);
        model.backend = ggml_backend_cpu_init();
    }

    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cpu_init() failed\n", __func__);
        return false;
    }

    // create the tensors for the model
    {
        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        model.wte     = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);
        model.wpe     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_ctx);
        model.lm_head = ggml_new_tensor_2d(ctx, wtype,         n_embd, n_vocab);

        // map by name
        model.tensors["model/ln_f/g"] = model.ln_f_g;
        model.tensors["model/ln_f/b"] = model.ln_f_b;

        model.tensors["model/wte"]     = model.wte;
        model.tensors["model/wpe"]     = model.wpe;
        model.tensors["model/lm_head"] = model.lm_head;

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            layer.ln_1_g        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_1_b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.ln_2_g        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);
            layer.ln_2_b        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_attn_attn_w = ggml_new_tensor_2d(ctx, wtype,           n_embd, 3*n_embd);
            layer.c_attn_attn_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3*n_embd);

            layer.c_attn_proj_w = ggml_new_tensor_2d(ctx, wtype,           n_embd, n_embd);
            layer.c_attn_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            layer.c_mlp_fc_w    = ggml_new_tensor_2d(ctx, wtype,           n_embd, 4*n_embd);
            layer.c_mlp_fc_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4*n_embd);

            layer.c_mlp_proj_w  = ggml_new_tensor_2d(ctx, wtype,         4*n_embd, n_embd);
            layer.c_mlp_proj_b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,   n_embd);

            // map by name
            model.tensors["model/h" + std::to_string(i) + "/ln_1/g"]        = layer.ln_1_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_1/b"]        = layer.ln_1_b;

            model.tensors["model/h" + std::to_string(i) + "/ln_2/g"]        = layer.ln_2_g;
            model.tensors["model/h" + std::to_string(i) + "/ln_2/b"]        = layer.ln_2_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/w"] = layer.c_attn_attn_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_attn/b"] = layer.c_attn_attn_b;

            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/w"] = layer.c_attn_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/attn/c_proj/b"] = layer.c_attn_proj_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/w"]    = layer.c_mlp_fc_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_fc/b"]    = layer.c_mlp_fc_b;

            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/w"]  = layer.c_mlp_proj_w;
            model.tensors["model/h" + std::to_string(i) + "/mlp/c_proj/b"]  = layer.c_mlp_proj_b;
        }
    }

    // allocate the model tensors in a backend buffer
    model.buffer_w = ggml_backend_alloc_ctx_tensors(ctx, model.backend);

    printf("%s: ggml tensor size    = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
    printf("%s: backend buffer size = %6.2f MB\n", __func__, ggml_backend_buffer_get_size(model.buffer_w)/(1024.0*1024.0));

    // override the default training context with the user-provided
    model.hparams.n_ctx = n_ctx;

    // key + value memory
    {
        auto * ctx = model.ctx_kv;

        // create the ggml context
        {
            size_t n_tensors = 2;
            struct ggml_init_params params = {
                /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ctx = ggml_init(params);
            if (!ctx) {
                fprintf(stderr, "%s: ggml_init() failed\n", __func__);
                return false;
            }
        }

        const auto & hparams = model.hparams;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;

        const int n_mem      = n_layer*n_ctx;
        const int n_elements = n_embd*n_mem;

        // k and v here can also be GGML_TYPE_F16 to save memory and speed up the computation
        // if backend supports it
        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        // allocate the KV memory in a backend buffer
        model.buffer_kv = ggml_backend_alloc_ctx_tensors(ctx, model.backend);

        const size_t memory_size = ggml_backend_buffer_get_size(model.buffer_kv);
        printf("%s: memory size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    // load weights
    {
        size_t total_size = 0;

        bool has_lm_head = false;

        std::vector<char> read_buf;

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = { 1, 1 };
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.c_str());
                return false;
            }

            auto tensor = model.tensors[name];
            ggml_set_name(tensor, name.c_str());
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.c_str());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.c_str(), (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            // for debugging
            if (0) {
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.c_str(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)), ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.c_str(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            if (ggml_backend_buffer_is_host(model.buffer_w)) {
                // for some backends such as CPU and Metal, the tensor data is in system memory and we can read directly into it
                fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(ggml_nbytes(tensor));
                fin.read(read_buf.data(), ggml_nbytes(tensor));
                ggml_backend_tensor_set(tensor, read_buf.data(), 0, ggml_nbytes(tensor));
            }

            // GPT-2 models share the WTE tensor as the LM head
            if (name == "model/wte" && has_lm_head == false) {
                //ggml_backend_tensor_copy(tensor, model.lm_head);
                model.lm_head = tensor;
            }

            if (name == "model/lm_head") {
                has_lm_head = true;
            }

            total_size += ggml_nbytes(tensor);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
    }

    fin.close();

    return true;
}

// build the computation graph
struct ggml_cgraph * gpt2_graph(
        const gpt2_model & model,
        const int n_past,
        const int n_tokens) {
    const int N = n_tokens;

    const auto & hparams = model.hparams;

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;

    // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
    static size_t buf_size = ggml_tensor_overhead()*GPT2_MAX_NODES + ggml_graph_overhead_custom(GPT2_MAX_NODES, false);
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };

    struct ggml_context * ctx = ggml_init(params);

    struct ggml_cgraph  * gf = ggml_new_graph_custom(ctx, GPT2_MAX_NODES, false);

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    // at this point, the tensor data is not allocated yet and cannot be set
    // we will find the tensor after the graph is allocated by its name, and set the data then
    ggml_set_name(embd, "embd");
    // setting a tensor as an input will ensure that it is allocated at the beginning of the graph
    // this is important to ensure that the input tensors are not overwritten before they are used
    ggml_set_input(embd);

    struct ggml_tensor * position = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
    ggml_set_name(position, "position");
    ggml_set_input(position);

    // wte + wpe
    struct ggml_tensor * inpL =
        ggml_add(ctx,
                ggml_get_rows(ctx, model.wte, embd),
                ggml_get_rows(ctx, model.wpe, position));

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor * cur;

        // norm
        {
            // [ 768, N]
            cur = ggml_norm(ctx, inpL, hparams.eps);

            // cur = ln_1_g*cur + ln_1_b
            // [ 768, N]
            cur = ggml_add(ctx,
                    ggml_mul(ctx,
                        cur,
                        model.layers[il].ln_1_g),
                    model.layers[il].ln_1_b);
        }

        // attn
        // [2304, 768] - model.layers[il].c_attn_attn_w
        // [2304,   1] - model.layers[il].c_attn_attn_b
        // [ 768,   N] - cur (in)
        // [2304,   N] - cur (out)
        //
        // cur = attn_w*cur + attn_b
        // [2304, N]
        {
            cur = ggml_mul_mat(ctx,
                    model.layers[il].c_attn_attn_w,
                    cur);

            cur = ggml_add(ctx,
                    cur,
                    model.layers[il].c_attn_attn_b);
        }

        // self-attention
        {
            struct ggml_tensor * Qcur = ggml_view_2d(ctx, cur, n_embd, N, cur->nb[1], 0*sizeof(float)*n_embd);
            struct ggml_tensor * Kcur = ggml_view_2d(ctx, cur, n_embd, N, cur->nb[1], 1*sizeof(float)*n_embd);
            struct ggml_tensor * Vcur = ggml_view_2d(ctx, cur, n_embd, N, cur->nb[1], 2*sizeof(float)*n_embd);

            // store key and value to memory
            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx, model.memory_v, N*n_embd, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

                ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            // [64, N, 12]
            struct ggml_tensor * Q =
                ggml_permute(ctx,
                        ggml_cont_3d(ctx, Qcur, n_embd/n_head, n_head, N),
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            // [64, n_past + N, 12]
            struct ggml_tensor * K =
                ggml_permute(ctx,
                        ggml_reshape_3d(ctx,
                            ggml_view_1d(ctx, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
                            n_embd/n_head, n_head, n_past + N),
                        0, 2, 1, 3);

            // GG: flash attention
            //struct ggml_tensor * V =
            //    ggml_cpy(ctx0,
            //            ggml_permute(ctx0,
            //                ggml_reshape_3d(ctx0,
            //                    ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
            //                    n_embd/n_head, n_head, n_past + N),
            //                1, 2, 0, 3),
            //            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_past + N, n_embd/n_head, n_head));

            //struct ggml_tensor * KQV = ggml_flash_attn(ctx0, Q, K, V, true);

            // K * Q
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ_scaled =
                ggml_scale(ctx,
                        KQ,
                        1.0f/sqrtf(float(n_embd)/n_head));

            // KQ_masked = mask_past(KQ_scaled)
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            // [n_past + N, N, 12]
            struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            // [n_past + N, 64, 12]
            struct ggml_tensor * V_trans =
                ggml_cont_3d(ctx,
                        ggml_permute(ctx,
                            ggml_reshape_3d(ctx,
                                ggml_view_1d(ctx, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
                                n_embd/n_head, n_head, n_past + N),
                            1, 2, 0, 3),
                        n_past + N, n_embd/n_head, n_head);

            // KQV = transpose(V) * KQ_soft_max
            // [64, N, 12]
            struct ggml_tensor * KQV = ggml_mul_mat(ctx, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            // [64, 12, N]
            struct ggml_tensor * KQV_merged = ggml_permute(ctx, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            // [768, N]
            cur = ggml_cont_2d(ctx, KQV_merged, n_embd, N);
        }

        // projection
        // [ 768, 768] - model.layers[il].c_attn_proj_w
        // [ 768,   1] - model.layers[il].c_attn_proj_b
        // [ 768,   N] - cur (in)
        // [ 768,   N] - cur (out)
        //
        // cur = proj_w*cur + proj_b
        // [768, N]
        {
            cur = ggml_mul_mat(ctx,
                    model.layers[il].c_attn_proj_w,
                    cur);

            cur = ggml_add(ctx,
                    cur,
                    model.layers[il].c_attn_proj_b);
        }

        // add the input
        cur = ggml_add(ctx, cur, inpL);

        struct ggml_tensor * inpFF = cur;

        // feed-forward network
        {
            // norm
            {
                cur = ggml_norm(ctx, inpFF, hparams.eps);

                // cur = ln_2_g*cur + ln_2_b
                // [ 768, N]
                cur = ggml_add(ctx,
                        ggml_mul(ctx,
                            cur,
                            model.layers[il].ln_2_g),
                        model.layers[il].ln_2_b);
            }

            // fully connected
            // [3072, 768] - model.layers[il].c_mlp_fc_w
            // [3072,   1] - model.layers[il].c_mlp_fc_b
            // [ 768,   N] - cur (in)
            // [3072,   N] - cur (out)
            //
            // cur = fc_w*cur + fc_b
            // [3072, N]
            cur = ggml_mul_mat(ctx,
                    model.layers[il].c_mlp_fc_w,
                    cur);

            cur = ggml_add(ctx,
                    cur,
                    model.layers[il].c_mlp_fc_b);

            // GELU activation
            // [3072, N]
            cur = ggml_gelu(ctx, cur);

            // projection
            // [ 768, 3072] - model.layers[il].c_mlp_proj_w
            // [ 768,    1] - model.layers[il].c_mlp_proj_b
            // [3072,    N] - cur (in)
            // [ 768,    N] - cur (out)
            //
            // cur = proj_w*cur + proj_b
            // [768, N]
            cur = ggml_mul_mat(ctx,
                    model.layers[il].c_mlp_proj_w,
                    cur);

            cur = ggml_add(ctx,
                    cur,
                    model.layers[il].c_mlp_proj_b);
        }

        // input for next layer
        inpL = ggml_add(ctx, cur, inpFF);
    }

    // norm
    {
        // [ 768, N]
        inpL = ggml_norm(ctx, inpL, hparams.eps);

        // inpL = ln_f_g*inpL + ln_f_b
        // [ 768, N]
        inpL = ggml_add(ctx,
                ggml_mul(ctx,
                    inpL,
                    model.ln_f_g),
                model.ln_f_b);
    }

    // inpL = WTE * inpL
    // [ 768, 50257] - model.lm_head
    // [ 768, N]     - inpL
    inpL = ggml_mul_mat(ctx, model.lm_head, inpL);
    ggml_set_name(inpL, "logits");
    // setting a tensor as the output will ensure that it is not overwritten by subsequent operations
    ggml_set_output(inpL);

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    ggml_build_forward_expand(gf, inpL);

    ggml_free(ctx);

    return gf;
}

// evaluate the transformer
//
//   - model:     the model
//   - allocr:    ggml_gallocr to use to allocate the compute buffer
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
bool gpt2_eval(
        const gpt2_model & model,
        ggml_gallocr_t allocr,
        const int n_threads,
        const int n_past,
        const std::vector<gpt_vocab::id> & embd_inp,
              std::vector<float>         & embd_w) {
    const int N = embd_inp.size();

    const auto & hparams = model.hparams;

    const int n_vocab = hparams.n_vocab;

    struct ggml_cgraph * gf = gpt2_graph(model, n_past, embd_inp.size());

    // allocate the graph tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    // set the graph inputs
    struct ggml_tensor * embd = ggml_graph_get_tensor(gf, "embd");
    ggml_backend_tensor_set(embd, embd_inp.data(), 0, N*ggml_element_size(embd));

    struct ggml_tensor * position = ggml_graph_get_tensor(gf, "position");
    for (int i = 0; i < N; ++i) {
        int32_t v = n_past + i;
        ggml_backend_tensor_set(position, &v, i*sizeof(int32_t), sizeof(v));
    }

    // set backend options
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    // run the computation
    ggml_backend_graph_compute(model.backend, gf);

    if (n_past%100 == 0) {
        ggml_graph_print   (gf);
        ggml_graph_dump_dot(gf, NULL, "gpt-2.dot");
    }

    // get the graph outputs
    struct ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");

    //embd_w.resize(n_vocab*N);
    //ggml_backend_tensor_get(logits, embd_w.data(), 0, sizeof(float)*n_vocab*N);

    // return result just for the last token
    embd_w.resize(n_vocab);
    ggml_backend_tensor_get(logits, embd_w.data(), (n_vocab*(N-1))*sizeof(float), sizeof(float)*n_vocab);

    return true;
}

/*
1. 目前是在 visual studio code 中编写的注释，在 visual studio 对 EditorConfig 支持不像 visual studio code 对 EditorConfig 支持那么好，所以有些配置项可能不起作用。因此，建议使用 visual studio code 来编辑和查看这些注释，以确保所有配置项都能正确应用。
2. 该项目中的文件使用 UTF-8 编码而不是 UTF-8 with BOM 编码。这是因为 UTF-8 with BOM 编码在某些编译器中可能会引起问题，尤其是在处理源代码文件时，但是在 MSVC 编译器工具链中则不会出现问题。因此，为了确保代码在大多数编译器工具链中的兼容性和可移植性，建议使用纯粹的 UTF-8 编码。对于 MSVC 编译器工具链，可以通过添加编译器选项 /utf-8 来启用对 UTF-8 编码的支持。
*/
int main(int argc, char ** argv) {
    /*
    1. ggml_time_init 函数是 ggml-base 目标提供给外部使用的一个时间初始化函数。该函数的作用是初始化时间相关的设置，以便后续的时间测量和计算能够准确进行。
    2. ggml_time_init 函数功能：
      2.1 Windows 下：读取高精度计时器频率与当前计数（QueryPerformanceFrequency/Counter），保存在静态 timer_freq/timer_start，供后续 ggml_time_ms/ggml_time_us 计算相对时间；通过减去程序启动时刻，降低频率与运行时间乘积溢出的风险。
      2.2 非 Windows 下：空实现，因为 ggml_time_ms/ggml_time_us 直接调用 clock_gettime(CLOCK_MONOTONIC)，无需预先初始化。
    3. 设计意图：在程序开始时调用一次，确保跨平台的高精度、单调计时接口可用，并尽量避免大 uptime 时的溢出。
    */
    ggml_time_init();

    /*
    1. t_main_start_us 变量用于记录主函数开始执行时的时间戳，单位为微秒（microseconds）。它通过调用 ggml_time_us() 函数获取当前的时间戳。
    */
    const int64_t t_main_start_us = ggml_time_us();

    /*
    1. 创建一个 gpt_params 结构体实例 params，用于存储 GPT-2 模型的参数配置。
    2. 设置模型文件路径为 "models/gpt-2-117M/ggml-model.bin"，该模型文件路径作为一个初始值，后续可能会被命令行参数覆盖。
    3. 调用 gpt_params_parse 函数解析命令行参数 argc 和 argv，将解析结果存储到 params 结构体中，该函数支持很多的命令行参数，可以查看具体的实现分析支持哪些参数。如果解析失败，程序将返回 1 并退出。
    */
    gpt_params params;
    params.model = "models/gpt-2-117M/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    /*
    1. 检查从命令行读取的随机种子参数 params.seed 是否小于 0。如果随机种子参数小于 0，则使用当前时间作为随机种子。该参数用来初始化生成器，后面不仅用于随机生成默认提示词，也用于采样过程即用于采样下一个 token，以确保生成的文本具有一定的随机性和多样性。
    2. 输出当前使用的随机种子值，方便调试和记录实验条件。
    */
    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    /*
    1. 在计算机科学中，伪随机数并非真正的随机，而是基于一个初始值（种子）通过特定算法计算出的一系列数字序列。这里创建一个 Mersenne Twister 随机数生成器 rng，并使用参数 params.seed 作为种子进行初始化。
    2. 检查参数 params.prompt 是否为空字符串。如果为空，则调用 gpt_random_prompt 函数生成一个随机的提示语，并将其赋值给 params.prompt。随机数生成器对随机提示语的生成、采样过程两方面产生影响。
      2.1 随机模式：通常作为默认设置。程序会读取当前系统时间（time(NULL)）作为种子。因为时间在不断流逝，每次运行程序时种子都不同，生成的文本也不同。
      2.2 复现模式：如果你在命令行显式传入一个整数（例如 42），程序将始终使用这个固定的种子。这对于调试非常重要，因为它能让你完美复现某一次的模型输出结果。
    */
    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }

    /*
    1. 创建变量 t_load_us，用于记录模型加载所花费的时间，单位为微秒（microseconds）。初始值为 0。
    2. 创建 gpt_vocab 结构体实例 vocab，用于存储 GPT-2 模型的词汇表。
    3. 创建 gpt2_model 结构体实例 model，用于存储 GPT-2 模型的参数和权重。
    */
    int64_t t_load_us = 0;

    gpt_vocab vocab;
    gpt2_model model;

    // load the model
    {
        /*
        1. 记录模型加载开始的时间戳，单位为微秒（microseconds）。调用 ggml_time_us() 函数获取当前时间戳，并将其存储在 t_start_us 变量中。
        2. 调用 gpt2_model_load 函数加载 GPT-2 模型。该函数接受模型文件路径 params.model、模型实例 model、词汇表实例 vocab、上下文长度参数 params.n_ctx 和 GPU 层数参数 params.n_gpu_layers 作为输入。
          2.1 如果模型加载失败（即 gpt2_model_load 函数返回 false），则输出错误信息并返回 1，结束程序执行。
          2.2 如果模型加载成功，计算模型加载所花费的时间，并将其存储在 t_load_us 变量中。计算方法是调用 ggml_time_us() 函数获取当前时间戳，并减去 t_start_us。
        3. 调用 test_gpt_tokenizer 函数测试 GPT-2 词汇表的分词功能，使用词汇表实例 vocab 和测试字符串 params.token_test 作为输入。
        */
        const int64_t t_start_us = ggml_time_us();

        if (!gpt2_model_load(params.model, model, vocab, params.n_ctx, params.n_gpu_layers)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;

        test_gpt_tokenizer(vocab, params.token_test);
    }

    /*
    1. 创建 ggml_gallocr_t 类型的变量 allocr，并初始化为 NULL。该变量将用于分配计算缓冲区。
    2. 分配计算缓冲区：
      2.1 创建一个图形分配器 allocr，使用模型后端的默认缓冲区类型进行初始化。
      2.2 计算最坏情况下的内存使用情况，以估计所需的计算缓冲区大小。具体步骤如下：
        2.2.1 计算令牌数量 n_tokens，取模型上下文长度 model.hparams.n_ctx 和参数 params.n_batch 的最小值。
        2.2.2 计算过去的令牌数量 n_past，等于模型上下文长度减去当前令牌数量 n_tokens。
        2.2.3 调用 gpt2_graph 函数构建计算图 gf，传入模型实例 model、过去的令牌数量 n_past 和当前令牌数量 n_tokens 作为参数。
      2.3 可选地，为最坏情况预分配计算缓冲区：
        2.3.1 调用 ggml_gallocr_reserve 函数，为图形分配器 allocr 预留计算图 gf 所需的内存。
    */
    ggml_gallocr_t allocr = NULL;
    // allocate the compute buffer
    {
        // create a graph allocator with the backend's default buffer type
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // create the worst case graph for memory usage estimation
        int n_tokens = std::min(model.hparams.n_ctx, params.n_batch);
        int n_past = model.hparams.n_ctx - n_tokens;
        struct ggml_cgraph * gf = gpt2_graph(model, n_past, n_tokens);

        // pre-allocate the compute buffer for the worst case (optional)
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size =  ggml_gallocr_get_buffer_size(allocr, 0);
        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0/1024.0);
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    /*
    1. 使用 gpt_tokenize 函数对输入提示语 params.prompt 进行分词，生成对应的令牌 ID 列表 embd_inp。该函数接受词汇表实例 vocab 和提示语字符串 params.prompt 作为输入。
    2. 更新参数 params.n_predict，确保预测的令牌数量不会超过模型的上下文长度限制。具体做法是将 params.n_predict 设置为其当前值和模型上下文长度减去输入令牌数量 embd_inp.size() 之间的较小值，如果传入的值过大，则进行调整。
    3. 输出提示语信息，包括提示语内容、提示语中的令牌数量以及前 8 个令牌的 ID，方便调试和记录实验条件。
    */
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu, first 8 tokens: ", __func__, embd_inp.size());
    for (int i = 0; i < std::min(8, (int) embd_inp.size()); i++) {
        printf("%d ", embd_inp[i]);
    }
    printf("\n\n");

    // submit the input prompt token-by-token
    // this reduces the memory usage during inference, at the cost of a bit of speed at the beginning
    std::vector<gpt_vocab::id> embd;

    /*
    1. 使用一个 for 循环来生成和输出预测的 token,直到达到指定的预测数量 params.n_predict，或者遇到结束标记（end-of-text token），之前已经将预测数量的边界情况处理好了。
    */
    for (size_t i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        /*
        1. 如果 embd 向量中的 token 数量大于 0，表示有新的 token 需要进行预测，使用 gpt2_eval 函数对当前的模型进行评估和预测。
        */
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!gpt2_eval(model, allocr, params.n_threads, n_past, embd, logits)) {
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        /*
        1. 检查当前是否已经处理完输入提示语 embd_inp 中的所有 token。
          1.1 如果已经处理完（即 i 大于等于 embd_inp.size()），则进行下一个 token 的采样：
            1.1.1 从参数 params 中获取 top_k、top_p 和温度 temp，用于控制采样过程。
            1.1.2 获取模型的词汇表大小 n_vocab。
            1.1.3 使用 gpt_sample_top_k_top_p 函数根据当前的 logits 和采样参数，从词汇表中采样下一个 token 的 ID，并将其存储在变量 id 中。
            1.1.4 记录采样所花费的时间，并累加到 t_sample_us 变量中。
            1.1.5 将采样得到的 token ID 添加到 embd 向量中，准备进行下一轮预测。
          1.2 如果还没有处理完输入提示语中的 token，则继续将提示语中的 token 添加到 embd 向量中，直到达到批处理大小 params.n_batch 为止。
        */
        if (i >= embd_inp.size()) {
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (size_t k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                if (int32_t(embd.size()) >= params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id : embd) {
            printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (!params.ignore_eos && embd.back() == 50256) {
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx_w);

    ggml_gallocr_free(allocr);
    ggml_backend_buffer_free(model.buffer_w);
    ggml_backend_buffer_free(model.buffer_kv);
    ggml_backend_free(model.backend);

    return 0;
}
