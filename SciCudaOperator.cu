#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "SciCudaOperator.hpp"

#define CHK_CUDASTATUS_AND_RETURN(cudaStatus, api)                      \
    if ((cudaStatus) != cudaSuccess) {                                  \
        printf("%s failed. %u(%s)\n", api, cudaStatus, cudaGetErrorName(cudaStatus));\
        return -1;                             \
    }

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

template <typename TYPE>
struct cuda_mat_t
{
    TYPE fx0;
    TYPE fy0;
    TYPE scalex;
    TYPE scaley;
};

__global__ void yuv420sp_mask_kernel_new(uint16_t width, uint16_t height, cudaSurfaceObject_t surf_y, cudaSurfaceObject_t surf_uv, uint16_t *boxes_ptr)
{
    uint16_t *mat_ptr = boxes_ptr + blockIdx.x * 4;
    uint16_t box_x = *mat_ptr;
    uint16_t box_y = *(mat_ptr + 1);
    uint16_t box_w = *(mat_ptr + 2);
    uint16_t box_h = *(mat_ptr + 3);

    // force to even number.
    box_x = box_x & (~0x0001U);
    box_y = box_y & (~0x0001U);
    box_w += box_x & 0x01;  // if need to align start point(box_x), broaden the box weight
    box_h += box_h & 0x01;  // if need to align start point(box_y), broaden the box height 
    box_w = (box_w + 1) & (~0x0001U);
    box_h = (box_h + 1) & (~0x0001U);

    // every thread process a quanternion(YYYYUV).
    uint16_t part_num = (box_w + 2 * blockDim.x - 1) / (blockDim.x * 2);

    for (uint16_t h = 0; h < box_h; h += 2)
    {
        for (uint16_t p = 0; p < part_num; ++p)
        {
            uint16_t offset_on_box = (p * blockDim.x + threadIdx.x) * 2;
            if (offset_on_box < box_w)
            {
                // Y component
                surf2Dwrite<uint8_t>(0, surf_y, box_x + offset_on_box, box_y, cudaBoundaryModeTrap);        // Y11
                surf2Dwrite<uint8_t>(0, surf_y, box_x + offset_on_box + 1, box_y, cudaBoundaryModeTrap);    // Y12
                surf2Dwrite<uint8_t>(0, surf_y, box_x + offset_on_box, box_y+1, cudaBoundaryModeTrap);      // Y21
                surf2Dwrite<uint8_t>(0, surf_y, box_x + offset_on_box + 1, box_y+1, cudaBoundaryModeTrap);  // Y22

                // UV component
                surf2Dwrite<uint8_t>(128, surf_uv, box_x + offset_on_box, box_y / 2, cudaBoundaryModeTrap);     // U
                surf2Dwrite<uint8_t>(128, surf_uv, box_x + offset_on_box + 1, box_y / 2, cudaBoundaryModeTrap); // V
            }
        }

        box_y += 2;
    }
}

int yuv420sp_mask_new(cudaSurfaceObject_t surf_y, cudaSurfaceObject_t surf_uv, size_t width, size_t height, size_t boxes_num, uint16_t *boxes_ptr, cudaStream_t stream)
{
    yuv420sp_mask_kernel_new<<<boxes_num, 32, 0, stream>>>(width, height, surf_y, surf_uv, boxes_ptr);
    return 0;

}

class MaskOperator : public SciCudaOperator {
  public:
    virtual ~MaskOperator() {
        checkRuntime(cudaStreamDestroy(stream));

    }

    bool Init(SciCudaOperator::Param param = {}) {
        auto dstWidth  = ALIGN_16(param.dstW);
        auto dstHeight = ALIGN_16(param.dstH);

        dstAttrs.planeColorFmts[0] = NvSciColor_Y8;
        dstAttrs.planeColorFmts[1] = NvSciColor_V8U8;
        dstAttrs.planeWidths[0]    = dstWidth;
        dstAttrs.planeHeights[0]   = dstHeight;
        dstAttrs.planeWidths[1]    = dstWidth/2;
        dstAttrs.planeHeights[1]   = dstHeight/2;
        dstAttrs.needCpuAccess     = true;
        dstAttrs.layout            = NvSciBufImage_BlockLinearType;
        dstAttrs.planeCount        = 2;
        dstAttrs.access_perm       = NvSciBufAccessPerm_ReadWrite;
        dstAttrs.lumaBaseAddressAlign = 256;
        dstAttrs.hasPlaneColorStds = true;
        dstAttrs.planeColorStds[0] = NvSciColorStd_REC601_ER;
        dstAttrs.planeColorStds[1] = NvSciColorStd_REC601_ER;
        dstAttrs.planeColorStds[2] = NvSciColorStd_REC601_ER;
        dstAttrs.scanType          = NvSciBufScan_ProgressiveType;
        dstAttrs.vprFlag           = false;
        dstAttrs.bufType           = NvSciBufType_Image;

        srcBuf.reserve(MAX_NUM_PACKETS);
        dstBuf.reserve(MAX_NUM_PACKETS);
        checkRuntime(cudaStreamCreate(&stream));

        d_img_.resize(img_size_);
        d_boxes_.resize(boxes_num_ * 4);
        boxes_.resize(boxes_num_ * 4);
        prepare_boxes<uint16_t>(&boxes_[0], boxes_num_, width_, height_);

        checkRuntime(
            cudaMemcpyAsync(&d_boxes_[0], &boxes_[0],
                            sizeof(boxes_[0]) * boxes_.size(),
                            cudaMemcpyHostToDevice,
                            stream)
            );


        return true;
    }

    virtual int Compose() override {
        static int idx = 0;

        cudaArray_t src_cudaArray[MAX_PLANE_COUNT];
        cudaArray_t dst_cudaArray[MAX_PLANE_COUNT];
        auto cudaStatus = cudaGetMipmappedArrayLevel(&src_cudaArray[0], src->cudaMem.plane[0].mipmappedArray, 0U);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel src_0");
        cudaStatus = cudaGetMipmappedArrayLevel(&src_cudaArray[1], src->cudaMem.plane[1].mipmappedArray, 0U);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel src_1");


        cudaStatus = cudaGetMipmappedArrayLevel(&dst_cudaArray[0], dst->cudaMem.plane[0].mipmappedArray, 0U);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel dst_0");
        cudaStatus = cudaGetMipmappedArrayLevel(&dst_cudaArray[1], dst->cudaMem.plane[1].mipmappedArray, 0U);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaGetMipmappedArrayLevel dst_1");

        width_ = (size_t)dst->cudaMem.attrs.planeWidths[0];
        height_ = (size_t)dst->cudaMem.attrs.planeHeights[0];

        cudaStatus = cudaMemcpy2DArrayToArray(*(&dst_cudaArray[0]), 0, 0,
                                              *((cudaArray_const_t *)&src_cudaArray[0]), 0, 0,
                                              width_,
                                              height_,
                                              cudaMemcpyDeviceToDevice);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaMemcpy2D for plane 0");

        cudaStatus = cudaMemcpy2DArrayToArray(*(&dst_cudaArray[1]), 0, 0,
                                              *((cudaArray_const_t *)&src_cudaArray[1]), 0, 0,
                                              width_,
                                              height_ / 2,
                                              cudaMemcpyDeviceToDevice);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaMemcpy2D for plane 1");

        cudaSurfaceObject_t dst_y;
        cudaResourceDesc surface_desc_y;
        memset(&surface_desc_y, 0, sizeof(cudaResourceDesc));
        surface_desc_y.resType = cudaResourceTypeArray;
        surface_desc_y.res.array.array =  *((cudaArray_t *)&dst_cudaArray[0]);
        cudaStatus = cudaCreateSurfaceObject(&dst_y, &surface_desc_y);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaCreateSurfaceObject surface_y");

        cudaSurfaceObject_t dst_uv;
        cudaResourceDesc surface_desc_uv;
        memset(&surface_desc_uv, 0, sizeof(cudaResourceDesc));
        surface_desc_uv.resType = cudaResourceTypeArray;
        surface_desc_uv.res.array.array =  *((cudaArray_t *)&dst_cudaArray[1]);
        cudaStatus = cudaCreateSurfaceObject(&dst_uv, &surface_desc_uv);
        CHK_CUDASTATUS_AND_RETURN(cudaStatus, "cudaCreateSurfaceObject surface_uv");

        // TODO
        yuv420sp_mask_new(dst_y, dst_uv, width_, height_, boxes_num_, &d_boxes_[0], stream);

        idx += 1;
        cudaStreamSynchronize(stream);

        return 0;
    }

    virtual int Compose(NvSciBufObj srcObj, NvSciBufObj dstObj = NULL) override {
        src = &srcBuf[(uint64_t)srcObj];
        dst = &dstBuf[(uint64_t)dstObj];

        if (src->obj == NULL) {
            src->obj = srcObj;
            src->mmap_cuda();
        }

        if (dst->obj == NULL) {
            if (dstObj) {
                dst->obj = dstObj;
            } else {
                // new dst.obj
                dst->alloc(&dstAttrs);
            }

            dst->mmap_cuda();
        }

        Compose();
        return 0;
    }


    template <typename T>
    void prepare_boxes(T *boxes_ptr, size_t num, size_t width, size_t height, size_t box_wight = 32, size_t box_heigh = 32)
    {
        size_t width_step = width / num;
        size_t height_step = height / num;

        for (size_t i = 0; i < num; ++i)
        {
            size_t offset = 4 * i;
            boxes_ptr[offset] = static_cast<uint16_t>(width_step * i); // ((width_step * i) / 2) * 2;  // force to even number
            boxes_ptr[offset + 1] = static_cast<uint16_t>(height_step * i); // ((height_step * i) / 2) * 2;  // force to even number
            boxes_ptr[offset + 2] = static_cast<uint16_t>(box_wight);
            boxes_ptr[offset + 3] = static_cast<uint16_t>(box_heigh);
        }
    }

    size_t width_     = 3840;
    size_t height_    = 2160;
    size_t img_size_  = width_*height_ * 3/2;
    size_t boxes_num_ = 20;
    size_t boxes_size = boxes_num_ * 4;
    size_t boxes_size_in_byte = boxes_size * sizeof(uint16_t);

    pinned_vector<uint16_t>  boxes_;
    device_vector<uint16_t>  d_boxes_;
    device_vector<uint8_t>   d_img_;
};


std::shared_ptr<SciCudaOperator> create_maskOperator(SciCudaOperator::Param param) {
    std::shared_ptr<MaskOperator> impl(new MaskOperator());
    if (!impl->Init(param)) {
        impl.reset();
    }
    return impl;
}
