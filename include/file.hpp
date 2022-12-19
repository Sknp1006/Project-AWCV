#pragma once
#ifndef H_AWCV_FILE
#define H_AWCV_FILE

#include <algorithm>
#include <boost/filesystem.hpp>
#include <iostream>

namespace awcv
{
//--------------------------------------------------------------------------------------------------------------------------------------
// 文件遍历器
//--------------------------------------------------------------------------------------------------------------------------------------
typedef std::vector<std::vector<std::string>> FilesVector; // 文件容器
class FileHandle
{
  public:
    FileHandle();
    FileHandle(std::string Path, std::string Extension, bool Recursion = false);
    ~FileHandle();

    void getFiles(std::string Path, std::string Extension, bool Recursion = false); // 获取文件
    FilesVector getFilesVector();

  private:
    FilesVector filesVector; // 文件列表
    static bool GreaterEqSort(std::string filePath1, std::string filePath2);
    static bool LessSort(std::string filePath1, std::string filePath2);
};
} // namespace awcv

#endif
