#include "read_file.h"

// 获取所有的文件名
void GetAllFiles(string path, vector<string>& files)
{
    std::cout << "Looking in directory: " << path << std::endl;  // 打印正在查找的目录
    struct dirent* entry;
    DIR* dp = opendir(path.c_str());
    if (dp == nullptr) {
        perror("opendir");
        return;
    }

    while ((entry = readdir(dp)) != nullptr) {
        string file_name = entry->d_name;
        if (entry->d_type == DT_DIR) { // 目录
            if (file_name != "." && file_name != "..") {
                std::cout << "Found directory: " << file_name << std::endl; // 打印找到的目录
                GetAllFiles(path + "/" + file_name, files); // 递归
            }
        } else { // 文件
            std::cout << "Found file: " << file_name << std::endl; // 打印找到的文件
            files.push_back(path + "/" + file_name);
        }
    }
    closedir(dp);
}

// 获取特定格式的文件名
void GetAllFormatFiles(string path, vector<string>& files, string format)
{
    std::cout << "Looking in directory: " << path << " for format: " << format << std::endl;  // 打印正在查找的目录和格式
    struct dirent* entry;
    DIR* dp = opendir(path.c_str());
    if (dp == nullptr) {
        perror("opendir");
        return;
    }

    while ((entry = readdir(dp)) != nullptr) {
        string file_name = entry->d_name;
        if (entry->d_type == DT_DIR) { // 目录
            if (file_name != "." && file_name != "..") {
                std::cout << "Found directory: " << file_name << std::endl; // 打印找到的目录
                GetAllFormatFiles(path + "/" + file_name, files, format); // 递归
            }
        } else if (file_name.find(format) != string::npos) { // 判断文件是否包含格式
            std::cout << "Found " << format << " file: " << file_name << std::endl; // 打印符合格式的文件
            files.push_back(path + "/" + file_name);
        }
    }
    closedir(dp);
}