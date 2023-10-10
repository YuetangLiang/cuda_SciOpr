# SciCudaOperator
Our source code (including libraries starting with the v2.x versions, and all versions), and ALL source code will be licensed under the [Elastic License v2 (ELv2)](https://www.elastic.co/licensing/elastic-license).

We chose ELv2 because of its permissiveness and simplicity. We're following the well-paved path of other great infrastructure projects like Elasticsearch and MongoDB that have implemented similar source code licenses to preserve their communities. Our community and customers still have no-charge and open access to use, modify, redistribute, and collaborate on the code. ELv2 also protects our continued investment in developing freely available libraries and developer tools by restricting cloud service providers from offering it as a service.

## sample
```cpp
std::shared_ptr<SciCudaOperator> maskOpr = create_maskOperator({3840, 2160});
m_maskOpr->Compose(pSciBufObj);
assert(m_maskOpr->dst && m_maskOpr->dst->obj);
```

## Create NvSciBufObj
```cpp
auto sciBufWidth  = ALIGN_16(3840);
auto sciBufHeight = ALIGN_16(2160);

BufObj sciBuf;
SurfaceAttrs sciBufAttrs{};

sciBufAttrs.planeColorFmts[0] = NvSciColor_Y8;
sciBufAttrs.planeColorFmts[1] = NvSciColor_V8U8;
sciBufAttrs.planeWidths[0]    = sciBufWidth;
sciBufAttrs.planeHeights[0]   = sciBufHeight;
sciBufAttrs.planeWidths[1]    = sciBufWidth/2;
sciBufAttrs.planeHeights[1]   = sciBufHeight/2;
sciBufAttrs.needCpuAccess     = true;
sciBufAttrs.layout            = NvSciBufImage_BlockLinearType;
sciBufAttrs.planeCount        = 2;
sciBufAttrs.access_perm       = NvSciBufAccessPerm_ReadWrite;
sciBufAttrs.lumaBaseAddressAlign = 256;
sciBufAttrs.hasPlaneColorStds = true;
sciBufAttrs.planeColorStds[0] = NvSciColorStd_REC601_ER;
sciBufAttrs.planeColorStds[1] = NvSciColorStd_REC601_ER;
sciBufAttrs.planeColorStds[2] = NvSciColorStd_REC601_ER;
sciBufAttrs.scanType          = NvSciBufScan_ProgressiveType;
sciBufAttrs.vprFlag           = false;
sciBufAttrs.bufType           = NvSciBufType_Image;

sciBuf.alloc(&sciBufAttrs);
assert(sciBuf.obj);
```
