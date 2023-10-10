# SciCudaOperator

## sample
```cpp
std::shared_ptr<SciCudaOperator> maskOpr = create_maskOperator({3840, 2160});
m_maskOpr->Compose(pSciBufObj);
assert(m_maskOpr->dst && m_maskOpr->dst->obj);
```
