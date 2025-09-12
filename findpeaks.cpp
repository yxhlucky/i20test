//
// Created by rfv56 on 2023/11/20.
//
#include "findpeaks.h"

vector<int> removePeaksBelowMinPeakHeight(vector<float> data, vector<int> ipk, float minph) {
    vector<int> ans;
    for (int i = 0; i < ipk.size(); ++i) {
        if (data[ipk[i]] > minph)
            ans.push_back(ipk[i]);
    }
    return ans;
}

vector<int> removePeaksBelowMinPeakHeight(vector<double> data, vector<int> ipk, double minph) {
    vector<int> ans;
    for (int i = 0; i < ipk.size(); ++i) {
        if (data[ipk[i]] > minph)
            ans.push_back(ipk[i]);
    }
    return ans;
}

vector<int> removePeaksBelowMinPeakProminence(vector<float> data, vector<int> ipk, float minpp) {
    vector<int> ans, base_loc;
    vector<float> prominence;
    for (int i = 0; i < ipk.size(); ++i) {  // get_prominence
        float min_right = 10000;
        int min_right_loc = 0;
        for (int j = ipk[i] + 1; j < data.size(); ++j) {  // 向右找波谷/边界
            if (data[j] <= data[ipk[i]]) {
                if (data[j] < min_right) {
                    min_right = data[j];
                    min_right_loc = j;
                }
            } else
                break;
        }
        float min_left = 10000;
        int min_left_loc = 0;
        for (int j = ipk[i] - 1; j >= 0; --j) {  // 向左找波谷/边界
            if (data[j] <= data[ipk[i]]) {
                if (data[j] < min_left) {
                    min_left = data[j];
                    min_left_loc = j;
                }
            } else
                break;
        }
        if (min_right > min_left) {
            base_loc.push_back(min_right_loc);
            prominence.push_back(data[ipk[i]] - min_right);
        } else {
            base_loc.push_back(min_left_loc);
            prominence.push_back(data[ipk[i]] - min_left);
        }
    }
    for (int i = 0; i < ipk.size(); ++i) {
        if (prominence[i] >= minpp) {
            ans.push_back(ipk[i]);
        }
    }
    return ans;
}

void insertion_sort(vector<float> &data, vector<int> &index, int left, int right) {
    int i, j;
    float temp;
    int temp_index;
    for (i = left + 1; i <= right; i++) {
        temp = data[i];
        temp_index = index[i];
        j = i - 1;
        while (j >= left && data[j] > temp) {
            data[j + 1] = data[j];
            index[j + 1] = index[j];
            j--;
        }
        data[j + 1] = temp;
        index[j + 1] = temp_index;
    }
}

void insertion_sort(vector<double> &data, vector<int> &index, int left, int right) {
    int i, j;
    double temp;
    int temp_index;
    for (i = left + 1; i <= right; i++) {
        temp = data[i];
        temp_index = index[i];
        j = i - 1;
        while (j >= left && data[j] > temp) {
            data[j + 1] = data[j];
            index[j + 1] = index[j];
            j--;
        }
        data[j + 1] = temp;
        index[j + 1] = temp_index;
    }
}


vector<int>
findPeaksSeparatedByMoreThanMinPeakDistance(vector<float> data, vector<int> ipk, float minpd) {
    vector<int> temp(ipk.size()), ans, ipk_copy;
    vector<float> peaks;
    ipk_copy.assign(ipk.begin(), ipk.end());
    reverse(ipk_copy.begin(), ipk_copy.end());
    for (int i = 0; i < ipk_copy.size(); ++i) {
        peaks.push_back(data[ipk_copy[i]]);
    }
    insertion_sort(peaks, ipk_copy, 0, peaks.size() - 1);
    reverse(ipk_copy.begin(), ipk_copy.end());
//    iota(temp.begin(), temp.end(), 0);
//    sort(temp.begin(), temp.end(),  // 按data的大小，给ipk_copy降序排序
//              [&peaks](size_t i1, size_t i2) { return peaks[i1] > peaks[i2]; });
//    for (int i = 0; i < temp.size(); ++i) {
//        ipk_copy.push_back(ipk[temp[i]]);
//    }
//    reverse(ipk_copy.begin(), ipk_copy.end());
//    for (int i = 0; i < ipk_copy.size(); ++i) {
//        cout << ipk_copy[i] << "  " << data[ipk_copy[i]] << endl;
//    }
    for (int i = 0; i < ipk_copy.size(); ++i) {
        if (i == 0) {
            ans.push_back(ipk_copy[i]);
        } else {
            bool flag = true;
            for (int j = 0; j < ans.size(); ++j) {
                if (abs(ipk_copy[i] - ans[j]) <= minpd)
                    flag = false;
            }
            if (flag)
                ans.push_back(ipk_copy[i]);
        }
    }
    sort(ans.begin(), ans.end());
    return ans;
}

vector<int>
findPeaksSeparatedByMoreThanMinPeakDistance(vector<double> data, vector<int> ipk, double minpd) {
    vector<int> temp(ipk.size()), ans, ipk_copy;
    vector<double> peaks;
    ipk_copy.assign(ipk.begin(), ipk.end());
    reverse(ipk_copy.begin(), ipk_copy.end());
    for (int i = 0; i < ipk_copy.size(); ++i) {
        peaks.push_back(data[ipk_copy[i]]);
    }
    insertion_sort(peaks, ipk_copy, 0, peaks.size() - 1);
    reverse(ipk_copy.begin(), ipk_copy.end());
    for (int i = 0; i < ipk_copy.size(); ++i) {
        if (i == 0) {
            ans.push_back(ipk_copy[i]);
        } else {
            bool flag = true;
            for (int j = 0; j < ans.size(); ++j) {
                if (abs(ipk_copy[i] - ans[j]) <= minpd)
                    flag = false;
            }
            if (flag)
                ans.push_back(ipk_copy[i]);
        }
    }
    sort(ans.begin(), ans.end());
    return ans;
}

vector<int> find_all_peaks(vector<float> data) {
    vector<int> ans;
    int start_peak = -1;
    int end_peak = -1;
    for (int i = 0; i < data.size(); i++) {
        if (i >= 1 && data[i] > data[i - 1])
            start_peak = i;
        if (i >= 1 && data[i] < data[i - 1] && start_peak >= 0) {
            end_peak = i - 1;
//            ans.push_back(floor((end_peak + start_peak) / 2.0));
            ans.push_back(start_peak);
            start_peak = -1;
        }
    }
    return ans;
}

vector<int> find_all_peaks(vector<double> data) {
    vector<int> ans;
    int start_peak = -1;
    int end_peak = -1;
    for (int i = 0; i < data.size(); i++) {
        if (i >= 1 && data[i] > data[i - 1])
            start_peak = i;
        if (i >= 1 && data[i] < data[i - 1] && start_peak >= 0) {
            end_peak = i - 1;
//            ans.push_back(floor((end_peak + start_peak) / 2.0));
            ans.push_back(start_peak);
            start_peak = -1;
        }
    }
    return ans;
}

vector<int> find_peaks(vector<float> data, float minph, float minpp, float minpd) {
    vector<int> all_peaks_loc, rm_minph_loc, rm_minpp_loc, rm_minpd_loc, ans;
    all_peaks_loc = find_all_peaks(data);
    rm_minph_loc = removePeaksBelowMinPeakHeight(data, all_peaks_loc, minph);
    rm_minpp_loc = removePeaksBelowMinPeakProminence(data, rm_minph_loc, minpp);
    rm_minpd_loc = findPeaksSeparatedByMoreThanMinPeakDistance(data, rm_minpp_loc, minpd);
    ans.assign(rm_minpd_loc.begin(), rm_minpd_loc.end());
    return ans;
}

vector<int> find_peaks(vector<double> data, double minpd) {
    vector<int> all_peaks_loc, rm_minpd_loc, ans;
    all_peaks_loc = find_all_peaks(data);
    rm_minpd_loc = findPeaksSeparatedByMoreThanMinPeakDistance(data, all_peaks_loc, minpd);
    ans.assign(rm_minpd_loc.begin(), rm_minpd_loc.end());
    return ans;
}

vector<int> find_peaks(vector<float> data, float minpd) {
    vector<int> all_peaks_loc, rm_minpd_loc, ans;
    all_peaks_loc = find_all_peaks(data);
    rm_minpd_loc = findPeaksSeparatedByMoreThanMinPeakDistance(data, all_peaks_loc, minpd);
    ans.assign(rm_minpd_loc.begin(), rm_minpd_loc.end());
    return ans;
}
vector<int> find_peaks(const vector<double>& data, double minpd, double minph) {
    vector<int> all_peaks_loc, rm_minph_loc, rm_minpd_loc, ans;
    all_peaks_loc = find_all_peaks(data);
    rm_minph_loc = removePeaksBelowMinPeakHeight(data, all_peaks_loc, minph);
    rm_minpd_loc = findPeaksSeparatedByMoreThanMinPeakDistance(data, rm_minph_loc, minpd);
    ans.assign(rm_minpd_loc.begin(), rm_minpd_loc.end());
    return ans;
}

vector<double> get_peaks(vector<double> data, double minpd) {
    vector<int> max_loc = find_peaks(data, minpd);
    // print max_loc
//    for (int i = 0; i < max_loc.size(); ++i) {
//        cout << max_loc[i] << "  " << data[max_loc[i]] << endl;
//    }
//    cout << max_loc << endl;
    vector<double> peaks;
    peaks.reserve(max_loc.size());
    for (int i = 0; i < max_loc.size(); ++i) {
        peaks.push_back(data[max_loc[i]]);
    }
    return peaks;
}

vector<float> get_peaks(vector<float> data, float minpd) {
    vector<int> max_loc = find_peaks(data, minpd);
    vector<float> peaks;
    peaks.reserve(max_loc.size());
    for (int i = 0; i < max_loc.size(); ++i) {
        peaks.push_back(data[max_loc[i]]);
    }
    return peaks;
}


vector<double> get_valley(vector<double> data, double minpd) {
    vector<double> temp;
    temp.assign(data.begin(), data.end());
    transform(temp.begin(), temp.end(), temp.begin(), [](double x) { return -x; });
    vector<int> min_loc = find_peaks(temp, minpd);
    vector<double> valley;
    valley.reserve(min_loc.size());
    for (int i = 0; i < min_loc.size(); ++i) {
        valley.push_back(data[min_loc[i]]);
    }
    return valley;
}

vector<float> get_valley(vector<float> data, float minpd) {
    vector<float> temp;
    temp.assign(data.begin(), data.end());
    transform(temp.begin(), temp.end(), temp.begin(), [](float x) { return -x; });
    vector<int> min_loc = find_peaks(temp, minpd);
    vector<float> valley;
    valley.reserve(min_loc.size());
    for (int i = 0; i < min_loc.size(); ++i) {
        valley.push_back(data[min_loc[i]]);
    }
    return valley;
}