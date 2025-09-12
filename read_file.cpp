#include "read_file.h"

// ��ȡ���е��ļ���
void GetAllFiles(string path, vector<string>& files)
{
    std::cout << "Looking in directory: " << path << std::endl;  // ��ӡ���ڲ��ҵ�Ŀ¼
    struct dirent* entry;
    DIR* dp = opendir(path.c_str());
    if (dp == nullptr) {
        perror("opendir");
        return;
    }

    while ((entry = readdir(dp)) != nullptr) {
        string file_name = entry->d_name;
        if (entry->d_type == DT_DIR) { // Ŀ¼
            if (file_name != "." && file_name != "..") {
                std::cout << "Found directory: " << file_name << std::endl; // ��ӡ�ҵ���Ŀ¼
                GetAllFiles(path + "/" + file_name, files); // �ݹ�
            }
        } else { // �ļ�
            std::cout << "Found file: " << file_name << std::endl; // ��ӡ�ҵ����ļ�
            files.push_back(path + "/" + file_name);
        }
    }
    closedir(dp);
}

// ��ȡ�ض���ʽ���ļ���
void GetAllFormatFiles(string path, vector<string>& files, string format)
{
    std::cout << "Looking in directory: " << path << " for format: " << format << std::endl;  // ��ӡ���ڲ��ҵ�Ŀ¼�͸�ʽ
    struct dirent* entry;
    DIR* dp = opendir(path.c_str());
    if (dp == nullptr) {
        perror("opendir");
        return;
    }

    while ((entry = readdir(dp)) != nullptr) {
        string file_name = entry->d_name;
        if (entry->d_type == DT_DIR) { // Ŀ¼
            if (file_name != "." && file_name != "..") {
                std::cout << "Found directory: " << file_name << std::endl; // ��ӡ�ҵ���Ŀ¼
                GetAllFormatFiles(path + "/" + file_name, files, format); // �ݹ�
            }
        } else if (file_name.find(format) != string::npos) { // �ж��ļ��Ƿ������ʽ
            std::cout << "Found " << format << " file: " << file_name << std::endl; // ��ӡ���ϸ�ʽ���ļ�
            files.push_back(path + "/" + file_name);
        }
    }
    closedir(dp);
}