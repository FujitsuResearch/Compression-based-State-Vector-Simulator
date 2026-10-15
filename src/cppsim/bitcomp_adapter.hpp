#pragma once

#if __has_include(<nvcomp/native/bitcomp.h>)
  #include <nvcomp/native/bitcomp.h>   // ← これが C Native API の本体（推奨）
#elif __has_include(<nvcomp/bitcomp.h>)
  #include <nvcomp/bitcomp.h>   // nvCOMP 側の bitcomp C API
#elif __has_include(<bitcomp.h>)
  #include <bitcomp.h>          // フラット配置
#else
  #error "bitcomp.h not found. Adjust include paths or install nvCOMP headers."
#endif
