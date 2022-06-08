# 033-multiprocess-parfor-python

033 multiprocess parfor python

parfor是matlab的多进程方案，对变量有限制，但是总体来说就是加个par即可，matlab优化很差，内存占用极多，所以parfor经常发生多核还不如单核的幽默结局。

python的优化明显好于matlab，并且句柄占用较少，cache使用情况较好，但是python的多进程不是像matlab一样加一个par就成的，这里就是使用map做的标准模板，套用即可。
