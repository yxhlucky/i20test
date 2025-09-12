#pragma once
#include <opencv2/opencv.hpp>

class MatCircularBuffer {
private:
    std::vector<cv::Mat> buffer;
    size_t capacity;
    size_t current_size;
    size_t head;
    size_t tail;

public:
    MatCircularBuffer(size_t cap) : capacity(cap), current_size(0), head(0), tail(0) {
        buffer.resize(capacity);
    }

    // ����е�ָ����Mat
    void pushColumn(const cv::Mat& col) {
        if (buffer[head].empty()) {
            col.copyTo(buffer[head]);
        }
        else {
            cv::hconcat(buffer[head], col, buffer[head]);
        }

        // �����ǰMat�������ƶ�headָ�뵽��һ��λ��
        if (buffer[head].cols >= capacity) {
            head = (head + 1) % capacity;
            if (head == tail) {
                tail = (tail + 1) % capacity;
                buffer[tail].release();  // �ͷ�������׼�������µ���
            }
            col.copyTo(buffer[head]);
        }

        if (current_size < capacity) {
            current_size++;
        }
    }

    //// ��ȡ�ض��е�Mat
    //cv::Mat getRows(const std::vector<int>& indices) const {
    //    cv::Mat result;
    //    for (size_t i = 0; i < current_size; i++) {
    //        cv::Mat selectedRows;
    //        for (int index : indices) {
    //            selectedRows.push_back(buffer[(tail + i) % capacity].row(index));
    //        }
    //        if (result.empty()) {
    //            selectedRows.copyTo(result);
    //        }
    //        else {
    //            cv::hconcat(result, selectedRows, result);
    //        }
    //    }
    //    return result;
    //}

    cv::Mat getRows(const std::vector<int>& indices) const {
        cv::Mat result;
        for (size_t i = 0; i < current_size; i++) {
            cv::Mat selectedRows;
            int total_rows = buffer[(tail + i) % capacity].rows;

            std::cout << "Processing buffer[" << i << "] with size: [" << total_rows << " x "
                << buffer[(tail + i) % capacity].cols << "], type: " << buffer[(tail + i) % capacity].type() << std::endl;

            for (int index : indices) {
                if (index < total_rows) {  // Check if the index is valid for this buffer
                    selectedRows.push_back(buffer[(tail + i) % capacity].row(index));
                    std::cout << "Successfully added row index " << index << " from buffer[" << i << "]" << std::endl;
                }
                else {
                    std::cout << "Invalid row index: " << index << " for buffer[" << i << "]. Total rows: " << total_rows << std::endl;
                }
            }
            if (result.empty()) {
                selectedRows.copyTo(result);
                std::cout << "Result was empty, copied selectedRows to result. Result size: [" << result.rows << " x " << result.cols << "]" << std::endl;
            }
            else {
                cv::hconcat(result, selectedRows, result);
                std::cout << "Concatenated selectedRows to result. Result size: [" << result.rows << " x " << result.cols << "]" << std::endl;
            }
        }
        return result;
    }



    // ��ȡ��ǰ�������Ĵ�С
    size_t size() const {
        return current_size;
    }

    // ��黺�����Ƿ�Ϊ��
    bool empty() const {
        return current_size == 0;
    }

    // ��ջ�����
    void clear() {
        for (size_t i = 0; i < capacity; ++i) {
            buffer[i].release();
        }
        head = tail = current_size = 0;
    }
};
