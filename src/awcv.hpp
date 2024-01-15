#pragma once
#ifndef AWCV_ALL_HPP
#define AWCV_ALL_HPP

#ifdef AWCV_CORE
#include "core.hpp"
#endif

#ifdef AWCV_DFT
#include "dft.hpp"
#endif

#ifdef AWCV_FACE
#include "face.hpp"
#endif

#ifdef AWCV_FILE
#include "file.hpp"
#endif

#ifdef AWCV_OFW
#include "opticalFlow.hpp"
#endif

#ifdef AWCV_REGION
#include "region.hpp"
#endif

#ifdef AWCV_SFM
#include "sfm.hpp"
#endif

#ifdef AWCV_MOIRE
#include "moire/moire.hpp"
#endif

#endif
