//
// Created by rfv56 on 2023/11/20.
//

#ifndef __FINDPEAKS_H
#define __FINDPEAKS_H

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace std;

vector<int> removePeaksBelowMinPeakHeight(vector<float> data, vector<int> ipk, float minph);

vector<int> removePeaksBelowMinPeakProminence(vector<float> data, vector<int> ipk, float minpp);

vector<int> findPeaksSeparatedByMoreThanMinPeakDistance(vector<float> data, vector<int> ipk, float minpd);

vector<int> findPeaksSeparatedByMoreThanMinPeakDistance(vector<double> data, vector<int> ipk, double minpd);

vector<int> find_all_peaks(vector<float> data);

vector<int> find_all_peaks(vector<double> data);

vector<int> find_peaks(vector<float> data, float minph, float minpp, float minpd);
vector<int> find_peaks(const vector<double>& data, double minpd, double minph);

vector<int> find_peaks(vector<double> data, double minpd);
vector<int> find_peaks(vector<float> data, float minpd);


vector<double> get_peaks(vector<double> data, double minpd);

vector<float> get_peaks(vector<float> data, float minpd);

vector<double> get_valley(vector<double> data, double minpd);

vector<float> get_valley(vector<float> data, float minpd);

#endif //__FINDPEAKS_H
