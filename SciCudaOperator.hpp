#pragma once
//#ifndef _SCICUDAOPERATOR_HPP_
#define _SCICUDAOPERATOR_HPP_

#include "nvmedia_6x/nvmedia_iep.h"
#include "nvmedia_6x/nvmedia_2d.h"
#include "nvmedia_6x/nvmedia_2d_sci.h"
#include "nvmedia_6x/nvmedia_nvscibuf.h"
#include "nvscibuf.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include <vector>
#include <utility>
#include <thread>
#include <string>
#include <set>
#include <memory>
#include <unordered_map>

#ifndef LOG_ERR
#define LOG_ERR printf
#endif

#define ALIGN_16(_x)  (uint16_t)(((_x) + static_cast<uint32_t>(15))  & (~static_cast<uint32_t>(15)))

#define checkRuntime(call)                                          \
    do {                                                            \
        check_runtime(call, #call, __LINE__, __FILE__);   \
    } while(0)

static inline bool check_runtime(cudaError_t e, const char *call, int lineno, const char *filename) {
    if (e != cudaSuccess) {
        fprintf(stderr,
                "CUDA Runtime error %s # %s, code = %s [ %d ] in file "
                "%s:%d\n",
                call, cudaGetErrorString(e), cudaGetErrorName(e), e, filename, lineno);

        abort();
        return false;
    }
    return true;
}


template <class T>
struct unified_memory_allocator {
    typedef std::size_t     size_type;
    typedef T   value_type;

    template <class U>
    unified_memory_allocator(unified_memory_allocator<U>&) noexcept {}
    unified_memory_allocator() noexcept = default;

    T* allocate(std::size_t n) {
        void* out = nullptr;
        checkRuntime(cudaMallocManaged(&out, n*sizeof(T), cudaMemAttachHost));
        printf("[UVM] alloc %p n:%lu\n", out, n);
        return static_cast<T*>(out);
    }

    void deallocate(T* p, std::size_t) noexcept {
        printf("[UVM] dealloc %p\n", reinterpret_cast<void*>(p));
        checkRuntime(cudaFree(p));
    }
};
template <class T>
struct pinned_memory_allocator {
    typedef std::size_t     size_type;
    typedef T   value_type;

    template <class U>
    pinned_memory_allocator(pinned_memory_allocator<U>&) noexcept {}
    pinned_memory_allocator() noexcept = default;

    T* allocate(std::size_t n) {
        void* out = nullptr;
        checkRuntime(cudaMallocHost(&out, n*sizeof(T)));
        printf("[pinned_memory] alloc %p n:(%lu x %lu)\n", out, n, sizeof(T));
        return static_cast<T*>(out);
    }

    void deallocate(T* p, std::size_t) noexcept {
        printf("[pinned_memory] dealloc %p\n", reinterpret_cast<void*>(p));
        checkRuntime(cudaFreeHost(p));
    }
};

template <class T>
struct device_memory_allocator {
    typedef std::size_t     size_type;
    typedef T   value_type;

    template <class U, class... Args> void construct(U*, Args&&...) {}

    template <class U>
    device_memory_allocator(device_memory_allocator<U>&) noexcept {}
    device_memory_allocator() noexcept = default;

    T* allocate(std::size_t n) {
        void* out = nullptr;
        checkRuntime(cudaMalloc(&out, n*sizeof(T)));
        printf("[device_memory] alloc %p n:(%lu x %lu)\n", out, n, sizeof(T));
        return static_cast<T*>(out);
    }

    void deallocate(T* p, std::size_t) noexcept {
        printf("[device_memory] dealloc %p\n", reinterpret_cast<void*>(p));
        checkRuntime(cudaFree(p));
    }
};

template <class T>
struct attr_allocator {
    typedef std::size_t     size_type;
    typedef T   value_type;
    NvSciBufModule bufModule = NULL;

    template <class U, class... Args> void construct(U*, Args&&...) {}

    template <class U>
    attr_allocator(attr_allocator<U>&) noexcept {}
    attr_allocator() noexcept = default;

    T* allocate(std::size_t n) {
        T* p = nullptr;
        auto e = NvSciBufModuleOpen(&bufModule);
        if(e != NvSciError_Success) {
            printf("NvSciBufModuleOpen failed with %d\n", e);
            return p;
        }

        p = new T[n];
        for(std::size_t i = 0U; i < n; i++) {
            e = NvSciBufAttrListCreate(bufModule, &p[i]);
            if (e != NvSciError_Success) {
                printf("NvSciBufAttrListCreate failed with %d\n", e);
                return p;
            }
        }

        printf("[NvSciBufAttrListCreate] alloc %p n:(%lu x %lu)\n", (void*)p, n, sizeof(T));
        return p;
    }

    void deallocate(T* p, std::size_t n) noexcept {
        if(bufModule == NULL) {
            return;
        }

        for(std::size_t i = 0U; i < n; i++) {
            NvSciBufAttrListFree(p[i]);
        }

        NvSciBufModuleClose(bufModule);
        bufModule = NULL;

        printf("[NvSciBufAttrListFree] dealloc %p n:(%lu x %lu)\n", (void*)p, n, sizeof(T));
        delete[] p;
    }
};

template <typename T>
using unified_vector = std::vector<T, unified_memory_allocator<T>>;
template <typename T>
using pinned_vector = std::vector<T, pinned_memory_allocator<T>>;
template <typename T>
using device_vector = std::vector<T, device_memory_allocator<T>>;
template <typename T>
using attr_vector = std::vector<T, attr_allocator<T>>;


/* Structure holding attributes related to a surface */
class SurfaceAttrs
{
 public:
    static constexpr size_t
        MAX_PLANE_COUNT = NV_SCI_BUF_IMAGE_MAX_PLANES; // defined in nvscibuf.h

    NvSciBufType                   bufType;// = NvSciBufType_Image;
    bool                           vprFlag;
    bool                           needCpuAccess;
    NvSciBufAttrValImageLayoutType layout;
    NvSciBufAttrValImageScanType scanType;
    uint32_t                        planeCount;
    NvSciBufAttrValAccessPerm       access_perm;
    uint32_t                        lumaBaseAddressAlign;
    NvSciBufAttrValColorFmt planeColorFmts[MAX_PLANE_COUNT];
    uint32_t planeWidths[MAX_PLANE_COUNT];
    uint32_t planeHeights[MAX_PLANE_COUNT];
    bool hasPlaneColorStds;
    NvSciBufAttrValColorStd planeColorStds[MAX_PLANE_COUNT];
    uint64_t   size;
    uint32_t   planePitches[MAX_PLANE_COUNT];
    uint32_t   planeBitsPerPixels[MAX_PLANE_COUNT];
    uint32_t   planeAlignedHeights[MAX_PLANE_COUNT];
    uint64_t   planeAlignedSizes[MAX_PLANE_COUNT];
    uint64_t   planeOffsets[MAX_PLANE_COUNT];
    uint8_t    planeChannelCounts[MAX_PLANE_COUNT];
    uint64_t   topPadding[MAX_PLANE_COUNT];
    uint64_t   bottomPadding[MAX_PLANE_COUNT];
    bool needSwCacheCoherency; //needCpuAccess

    void print() {
        for (auto i = 0U; i < planeCount; i++) {
            printf("[plane:%u, fmt:%u]: %ux%u, Aligned:%ux%ux(%2u bits) = %lu bytes, planeOffsets:%8lu, planeChannelCount:%u \n",
                i, planeColorFmts[i],
                planeWidths[i], planeHeights[i],
                planePitches[i], planeAlignedHeights[i],
                planeBitsPerPixels[i],
                planeAlignedSizes[i],
                planeOffsets[i],
                planeChannelCounts[i]);
        }
    }

    int getAttrs(NvSciBufObj obj) {
        NvSciBufAttrList att;
        NvSciBufObjGetAttrList(obj, &att);
        return getAttrs(att);
    }

    // 1st: NvSciBufObjGetAttrList(sciBufObj, &bufAttrList);
    int getAttrs(NvSciBufAttrList bufAttrList) {
        // PopulateBufAttr

        std::vector<NvSciBufAttrKeyValuePair> bAttr {
            {NvSciBufImageAttrKey_Size, NULL, 0},                      // 0
            {NvSciBufImageAttrKey_Layout, NULL, 0},                    // 1
            {NvSciBufImageAttrKey_PlaneCount, NULL, 0},                // 2
            {NvSciBufImageAttrKey_PlaneWidth, NULL, 0},                // 3
            {NvSciBufImageAttrKey_PlaneHeight, NULL, 0},               // 4
            {NvSciBufImageAttrKey_PlanePitch, NULL, 0},                // 5
            {NvSciBufImageAttrKey_PlaneBitsPerPixel, NULL, 0},         // 6
            {NvSciBufImageAttrKey_PlaneAlignedHeight, NULL, 0},        // 7
            {NvSciBufImageAttrKey_PlaneAlignedSize, NULL, 0},          // 8
            {NvSciBufImageAttrKey_PlaneChannelCount, NULL, 0},         // 9
            {NvSciBufImageAttrKey_PlaneOffset, NULL, 0},               // 10
            {NvSciBufImageAttrKey_PlaneColorFormat, NULL, 0},          // 11
            {NvSciBufImageAttrKey_TopPadding, NULL, 0},                // 12
            {NvSciBufImageAttrKey_BottomPadding, NULL, 0},             // 13
            {NvSciBufGeneralAttrKey_CpuNeedSwCacheCoherency, NULL, 0}  // 14
        };
        auto err = NvSciBufAttrListGetAttrs(bufAttrList, &bAttr[0], bAttr.size());
        if (err != NvSciError_Success) {
            return 1;
        }

        size = *(uint64_t*)bAttr[0].value;
        layout = *(NvSciBufAttrValImageLayoutType*)bAttr[1].value;
        planeCount = *(uint32_t*)bAttr[2].value;
        needSwCacheCoherency = *(bool*)bAttr[14].value;
#define PLANESIZE(its) (planeCount * sizeof(its[0]))
        memcpy(planeWidths,         bAttr[3].value,  PLANESIZE(planeWidths));
        memcpy(planeHeights,        bAttr[4].value,  PLANESIZE(planeHeights));
        memcpy(planePitches,        bAttr[5].value,  PLANESIZE(planePitches));
        memcpy(planeBitsPerPixels,  bAttr[6].value,  PLANESIZE(planeBitsPerPixels));
        memcpy(planeAlignedHeights, bAttr[7].value,  PLANESIZE(planeAlignedHeights));
        memcpy(planeAlignedSizes,   bAttr[8].value,  PLANESIZE(planeAlignedSizes));
        memcpy(planeChannelCounts,  bAttr[9].value,  PLANESIZE(planeChannelCounts));
        memcpy(planeOffsets,        bAttr[10].value, PLANESIZE(planeOffsets));
        memcpy(planeColorFmts,      bAttr[11].value, PLANESIZE(planeColorFmts));
        memcpy(topPadding,          bAttr[12].value, PLANESIZE(topPadding));
        memcpy(bottomPadding,       bAttr[13].value, PLANESIZE(bottomPadding));

        print();
        return 0;
    }

    int setCudaAttrs(NvSciBufAttrList cudaBufAttrList, int cudaDevId = 0) {
        cudaSetDevice(cudaDevId);
        CUuuid uuid;
        CUresult e = cuDeviceGetUuid(&uuid, cudaDevId);
        if (e != CUDA_SUCCESS) {
            return 1;
        }
        NvSciRmGpuId gpuId;
        memcpy(gpuId.bytes, uuid.bytes, sizeof(uuid.bytes));

        std::vector<NvSciBufAttrKeyValuePair> cudaBufAttr {
            {NvSciBufGeneralAttrKey_GpuId,             &gpuId, sizeof(gpuId)},
            {NvSciBufGeneralAttrKey_RequiredPerm,      &access_perm, sizeof(access_perm)},
            {NvSciBufGeneralAttrKey_Types,             &bufType, sizeof(bufType)},
            {NvSciBufGeneralAttrKey_NeedCpuAccess,     &needCpuAccess, sizeof(needCpuAccess)},
        };
        auto err = NvSciBufAttrListSetAttrs(cudaBufAttrList,
                                        &cudaBufAttr[0],
                                        cudaBufAttr.size());
        if (err != NvSciError_Success) {
            return 1;
        }

        return 0;
    }

    int set2DAttrs(NvSciBufAttrList nv2DBufAttrList, void* nv2D) {
        NvMediaStatus e = NvMedia2DFillNvSciBufAttrList(static_cast<NvMedia2D*>(nv2D), nv2DBufAttrList);
        if (e != NVMEDIA_STATUS_OK) {
            return 1;
        }

        std::vector<NvSciBufAttrKeyValuePair> nv2DBufAttr {
            {NvSciBufGeneralAttrKey_Types,             &bufType, sizeof(bufType)},
        };
        auto err = NvSciBufAttrListSetAttrs(nv2DBufAttrList,
                                        &nv2DBufAttr[0],
                                        nv2DBufAttr.size());
        if (err != NvSciError_Success) {
            return 1;
        }

        return 0;
    }

    int setIEPAttrs(NvSciBufAttrList iepBufAttrList, int encDevId = 0) {
        NvMediaStatus e = NvMediaIEPFillNvSciBufAttrList(
            static_cast<NvMediaEncoderInstanceId>(encDevId), iepBufAttrList);
        if (e != NVMEDIA_STATUS_OK) {
            return 1;
        }
        uint32_t baseAddrAlign[MAX_PLANE_COUNT] = {};
        uint64_t padding[MAX_PLANE_COUNT] = {};
        baseAddrAlign[0] = lumaBaseAddressAlign;
        baseAddrAlign[1] = lumaBaseAddressAlign;
        baseAddrAlign[2] = lumaBaseAddressAlign;
        std::vector<NvSciBufAttrKeyValuePair> iepBufAttr {
            {NvSciBufGeneralAttrKey_RequiredPerm,      &access_perm, sizeof(access_perm)},
            {NvSciBufGeneralAttrKey_Types,             &bufType, sizeof(bufType)},
            {NvSciBufGeneralAttrKey_NeedCpuAccess,     &needCpuAccess, sizeof(needCpuAccess)},
            {NvSciBufGeneralAttrKey_EnableCpuCache,    &needCpuAccess, sizeof(needCpuAccess)},
            {NvSciBufImageAttrKey_TopPadding,          &padding, planeCount * sizeof(padding[0])},
            {NvSciBufImageAttrKey_BottomPadding,       &padding, planeCount * sizeof(padding[0])},
            {NvSciBufImageAttrKey_LeftPadding,         &padding, planeCount * sizeof(padding[0])},
            {NvSciBufImageAttrKey_RightPadding,        &padding, planeCount * sizeof(padding[0])},
            {NvSciBufImageAttrKey_Layout,              &layout, sizeof(layout)},
            {NvSciBufImageAttrKey_PlaneCount,          &planeCount, sizeof(planeCount)},
            {NvSciBufImageAttrKey_PlaneColorFormat,    &planeColorFmts, planeCount * sizeof(NvSciBufAttrValColorFmt)},
            {NvSciBufImageAttrKey_PlaneColorStd,       &planeColorStds, planeCount * sizeof(NvSciBufAttrValColorStd)},
            {NvSciBufImageAttrKey_PlaneBaseAddrAlign,  &baseAddrAlign, planeCount * sizeof(uint32_t)},
            {NvSciBufImageAttrKey_PlaneWidth,          &planeWidths, planeCount * sizeof(uint32_t)},
            {NvSciBufImageAttrKey_PlaneHeight,         &planeHeights, planeCount * sizeof(uint32_t)},
            {NvSciBufImageAttrKey_VprFlag,             &vprFlag, sizeof(vprFlag)},
            {NvSciBufImageAttrKey_ScanType,            &scanType, sizeof(NvSciBufAttrValImageScanType)}
        };
        auto err = NvSciBufAttrListSetAttrs(iepBufAttrList,
                                        &iepBufAttr[0],
                                        iepBufAttr.size());
        if (err != NvSciError_Success) {
            return 1;
        }

        return 0;
    }
};




/* Structure holding allocated objects related to an NvSciBufObj */
class BufObj
{
 public:

    struct CudaMM {
        struct ExtMemPlane {
            cudaMipmappedArray_t mipmappedArray;
            struct cudaExternalMemoryMipmappedArrayDesc mipmapDesc;

            cudaError_t getMipmappedArray(cudaExternalMemory_t extMem, size_t w, size_t h, NvSciBufAttrValColorFmt fmt, uint64_t planeOffset, 
                                            size_t depth = 1, unsigned int flags = 0, unsigned int numLevels = 1) {
                mipmapDesc.flags     = flags;
                mipmapDesc.numLevels = numLevels;
                mipmapDesc.offset    = planeOffset;
                mipmapDesc.extent.width  = w;
                mipmapDesc.extent.height = h;
                mipmapDesc.extent.depth  = depth;
                if (fmt == NvSciColor_Y8) {
                    mipmapDesc.formatDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
                } else if (fmt <= NvSciColor_V8_U8) {
                    mipmapDesc.formatDesc = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
                } else {
                    printf("fmt: %u NOT support, switch to: UV \n", fmt);
                    mipmapDesc.formatDesc = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
                }

                return cudaExternalMemoryGetMappedMipmappedArray(&mipmappedArray, extMem, &mipmapDesc);
            }
        };
        std::vector<ExtMemPlane> plane;
        SurfaceAttrs  attrs;
        cudaExternalMemory_t extMem;

        void munmap() {
            for(auto &plane_:plane) {
                cudaFreeMipmappedArray(plane_.mipmappedArray);
            }
            cudaDestroyExternalMemory(extMem);
        }

        void *mmap(NvSciBufObj obj) {
            attrs.getAttrs(obj);

            cudaExternalMemoryHandleDesc desc{};
            desc.type = cudaExternalMemoryHandleTypeNvSciBuf;
            desc.handle.nvSciBufObject = obj;
            desc.size = attrs.size;
            auto cudaStatus = cudaImportExternalMemory(&extMem, &desc);
            if (cudaStatus != cudaSuccess) {
                return NULL;
            }

            if (attrs.layout != NvSciBufImage_BlockLinearType) {
                return NULL;
            }

            plane.resize(attrs.planeCount);
            for(auto i = 0U; i < plane.size(); i++) {
                auto e = plane[i].getMipmappedArray(extMem, 
                                            attrs.planeWidths[i], attrs.planeHeights[i], 
                                            attrs.planeColorFmts[i], attrs.planeOffsets[i]);
                if (e != cudaSuccess) {
                    return NULL;
                }
            }

            return NULL;
        }

    } cudaMem;

    SurfaceAttrs     attrs;
    NvSciBufAttrList attrListReconciled = NULL;

    NvSciBufObj      obj = NULL;
    NvMediaRect     *srcRect = NULL;
    NvMediaRect     *dstRect = NULL;
    NvMedia2DTransform transform = NVMEDIA_2D_TRANSFORM_NONE;


    void dealloc() {
        if(obj) {
            NvSciBufObjFree(obj);
            obj = NULL;
        }
        if (attrListReconciled) {
            NvSciBufAttrListFree(attrListReconciled);
            attrListReconciled = NULL;
        }
    }

    int alloc(SurfaceAttrs *att, void* nv2D = NULL, int encDevId = 0, int cudaDevId = 0) {
        NvSciError err = NvSciError_Success;
        attr_vector<NvSciBufAttrList> lists;
        lists.reserve(3);
        /*{
            buf->iepBufAttrList,
            buf->cudaBufAttrList,
            buf->nv2DBufAttrList,
            };*/

        lists.resize(2);
        if(nv2D) {
            lists.resize(3);
            att->set2DAttrs(lists[2], nv2D);
        }
        att->setCudaAttrs(lists[1], cudaDevId);
        att->setIEPAttrs(lists[0], encDevId);

        NvSciBufAttrList attrListConflict;
        err = NvSciBufAttrListReconcile(&lists[0],
                                        lists.size(),
                                        &attrListReconciled,
                                        &attrListConflict);
        if (err != NvSciError_Success) {
            return 1;
        }

        err = NvSciBufObjAlloc(attrListReconciled, &obj);
        if (err != NvSciError_Success) {
            return 1;
        }

        NvSciBufAttrListFree(attrListConflict);

        attrs = *att;
        return 0;
    }

    void *mmap_cuda() {
        return cudaMem.mmap(obj);
    }

    void munmap_cuda() {
        cudaMem.munmap();
    }
};

class SciCudaOperator {
 public:
    static constexpr uint32_t MAX_NUM_PACKETS = 6U;
    static constexpr size_t
        MAX_PLANE_COUNT = NV_SCI_BUF_IMAGE_MAX_PLANES; // defined in nvscibuf.h

    struct Param {
        uint32_t dstW;
        uint32_t dstH;
    };

    virtual ~SciCudaOperator() = default;
    virtual int Compose() = 0;
    virtual int Compose(NvSciBufObj srcObj, NvSciBufObj dstObj = NULL) = 0;

    std::unordered_map<uint64_t, BufObj> srcBuf;
    std::unordered_map<uint64_t, BufObj> dstBuf;
    BufObj *src = NULL;
    BufObj *dst = NULL;
    SurfaceAttrs dstAttrs{};
    cudaStream_t stream;
};

std::shared_ptr<SciCudaOperator> create_maskOperator(SciCudaOperator::Param param);


//#endif // _SCICUDAOPERATOR_HPP_
