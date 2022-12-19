#include "file.hpp"
//--------------------------------------------------------------------------------------------------------------------------------------
//												FileHandle构造函数
//--------------------------------------------------------------------------------------------------------------------------------------
awcv::FileHandle::FileHandle()
{
}
awcv::FileHandle::FileHandle(std::string Path, std::string Extension, bool Recursion)
{
    this->getFiles(Path, Extension, Recursion);
}
//--------------------------------------------------------------------------------------------------------------------------------------
//												FileHandle析构函数
//--------------------------------------------------------------------------------------------------------------------------------------
awcv::FileHandle::~FileHandle()
{
    FileHandle::filesVector.clear();
}
//--------------------------------------------------------------------------------------------------------------------------------------
//功能:获取文件列表
//参数:
//          Extension:          文件扩展名
//          Traverse:           是否遍历子文件夹
//返回值:
//          FilesVector:        std::vector<std::vector<std::string>>
//--------------------------------------------------------------------------------------------------------------------------------------
void awcv::FileHandle::getFiles(std::string Path, std::string Extension, bool Recursion)
{
    std::vector<std::string> files;                                                                 //文件容器
    std::vector<std::string> folder;                                                                //文件夹容器  
    boost::filesystem::path Root(Path);
    if (!boost::filesystem::exists(Root))
    {
        printf("The file path doesn't exist: %s", Root.string().c_str());                           //文件路径不存在
    }
    boost::filesystem::directory_iterator endIter;
    for (boost::filesystem::directory_iterator Iter(Root); Iter != endIter; Iter++)                 //文件迭代器
    {
        if (boost::filesystem::is_directory(*Iter)) 
        {
            if (Recursion)
            {
                folder.push_back((*Iter).path().string());                                          //待遍历文件夹
            }
        }
        else if (boost::filesystem::is_regular_file(*Iter) && Iter->path().extension() == Extension)
        {
            files.push_back((*Iter).path().string());
        }
    }
    if (folder.size() > 0)
    {
        if (folder.size() > 1) std::sort(folder.begin(), folder.end(), FileHandle::LessSort);       //文件名排序
        for (std::vector<std::string>::iterator folder_iter = folder.begin(); folder_iter != folder.end(); folder_iter++)
        {
            getFiles(*folder_iter, Extension, Recursion);
        }
    }
    if (files.size() > 0)
    {
        if (files.size() > 1) std::sort(files.begin(), files.end(), FileHandle::LessSort);          //文件名排序
        FileHandle::filesVector.push_back(files);                                                   //保存结果
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------
//功能:获取文件列表
//--------------------------------------------------------------------------------------------------------------------------------------
awcv::FilesVector awcv::FileHandle::getFilesVector()
{
    return this->filesVector;
}
//--------------------------------------------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------------------------------------------
bool awcv::FileHandle::GreaterEqSort(std::string filePath1, std::string filePath2)
{
    int len1 = filePath1.length();
    int len2 = filePath2.length();
    if (len1 < len2)
    {
        return false;
    }
    else if (len1 > len2)
    {
        return true;
    }
    else
    {
        int iter = 0;
        while (iter < len1)
        {
            if (filePath1.at(iter) < filePath2.at(iter))
            {
                return false;
            }
            else if (filePath1.at(iter) > filePath2.at(iter))
            {
                return true;
            }
            ++iter;
        }
    }
    return true;
}
//--------------------------------------------------------------------------------------------------------------------------------------
//
//--------------------------------------------------------------------------------------------------------------------------------------
bool awcv::FileHandle::LessSort(std::string filePath1, std::string filePath2)
{
    return (!FileHandle::GreaterEqSort(filePath1, filePath2));
}
