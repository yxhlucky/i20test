#pragma once
#include <iostream>
#include <dirent.h>  // macOS上用来替代 <io.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
using namespace std;

void GetAllFiles(string path, vector<string>& files);

void GetAllFormatFiles(string path, vector<string>& files, string format);