#pragma once

#include <sstream>
#include <iostream>
#include <complex>

#include <glog/logging.h>

#include "cuda_helper.h"

using namespace std;

// (default) SINGLE GPU
#define SINGLE_GPU

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&) = delete;\
  classname& operator=(const classname&) = delete;
