#include "main.h"
cv::Mat rotateImage(const cv::Mat& src, double angle)
{
    cv::Point2f center(src.cols / 2.0, src.rows / 2.0);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat rotated_image;
    cv::warpAffine(src, rotated_image, rotation_matrix, src.size(), cv::INTER_LINEAR);
    return rotated_image;
}

// 输入：旋转图里的 Rect + 旋转角度 + 原图尺寸
// 返回：原图坐标系的 Rect
cv::Rect mapBackToOriginal(const cv::Rect& rRot, double angle, const cv::Size& origSize)

{
    // 1. 构造与 rotateImage() 里完全一样的矩阵
    cv::Point2f c(origSize.width * 0.5f, origSize.height * 0.5f);
    cv::Mat M = cv::getRotationMatrix2D(c, angle, 1.0);   // 正变换

    // 2. 求逆矩阵（用于把坐标映射回去）
    cv::Mat M_inv; cv::invertAffineTransform(M, M_inv);

    // 3. 旋转图里矩形的 4 个角
    std::vector<cv::Point2f> corners = {
            {static_cast<float>(rRot.x),                    static_cast<float>(rRot.y)},
            {static_cast<float>(rRot.x + rRot.width),       static_cast<float>(rRot.y)},
            {static_cast<float>(rRot.x + rRot.width),       static_cast<float>(rRot.y + rRot.height)},
            {static_cast<float>(rRot.x),                    static_cast<float>(rRot.y + rRot.height)}
    };

    // 4. 逆变换回原图坐标
    std::vector<cv::Point2f> origCorners(4);
    cv::transform(corners, origCorners, M_inv);

    // 5. 取包围盒
    return cv::boundingRect(origCorners);
}
cv::Rect adjust_rect(const cv::Rect &currentRect, const cv::Rect &detectedFaceRect, const cv::Size &boundarySize) {
    // 计算当前矩形和检测到的人脸矩形的中心点
    cv::Point currentCenter = (currentRect.br() + currentRect.tl()) * 0.5;
    cv::Point detectedCenter = (detectedFaceRect.br() + detectedFaceRect.tl()) * 0.5;

    // 确定更新阈值（例如，当前矩形大小的10%）
    int thresholdX = currentRect.width * 0.1;
    int thresholdY = currentRect.height * 0.1;

    // 检查是否需要更新位置或大小
    if (std::abs(currentCenter.x - detectedCenter.x) > thresholdX ||
        std::abs(currentCenter.y - detectedCenter.y) > thresholdY ||
        std::abs(currentRect.width - detectedFaceRect.width) > thresholdX ||
        std::abs(currentRect.height - detectedFaceRect.height) > thresholdY ||
        currentRect.empty()) {
        // 扩大当前矩形的尺寸
        int newWidth = static_cast<int>(detectedFaceRect.width * 1);
        int newHeight = static_cast<int>(detectedFaceRect.height * 1);

        // 根据新的尺寸计算调整后的矩形位置
        int x = detectedCenter.x - newWidth / 2;
        int y = detectedCenter.y - newHeight / 2;

        // 确保新的宽度和高度为偶数
        if (newWidth % 2 != 0) newWidth -= 1;
        if (newHeight % 2 != 0) newHeight -= 1;

        // 确保x和y为偶数
        if (x % 2 != 0) x -= 1;
        if (y % 2 != 0) y -= 1;

        // 创建新的矩形框，并确保其在图像边界内
        cv::Rect dst_rect = cv::Rect(x, y, newWidth, newHeight);
        cv::Rect boundaryRect = cv::Rect(0, 0, boundarySize.width, boundarySize.height);

        // 确保新的矩形框大部分在边界内
        if ((dst_rect & boundaryRect).area() > 0.2 * dst_rect.area())
            return dst_rect & boundaryRect;
        else
            return cv::Rect(0, 0, 0, 0);
    }

    // 否则返回当前矩形
    return currentRect;
}


int main()
{
    //std::string folder = "/Users/yangxh/Data/i20_Data/ppg/";
    //std::string folder = "/Volumes/637data_expansion/yxh/i20/i20_data/ppg_bre/syy_ppg_bre/";
     std::string folder = "/Users/yangxh/Data/i20_Data/sean/";
    //std::string folder = "/Users/yangxh/Data/i20_Data/0617/wwj/";
   // std::string folder = "/Users/yangxh/Data/i20_Data/0618/yxh_20/";
    //std::string folder = "/Users/yangxh/Data/i20_Data/lily_baby_0624/10min_1.1m/";
    //std::string folder = "/Users/yangxh/Data/i20_Data/0624/yxh_nir/";
    //std::string folder = "/Users/yangxh/Data/i20_Data/0618/yxh_20/";
    //std::string folder = "/Users/yangxh/Data/i20_Data/0719/";

    int start_frame = 0;
    cv::Rect lastFaceBox;   // 上一次有效的人脸框
    cv::Rect currentFaceBox;  // 当前帧的人脸框
    //bool loadFromFile = true;
    std::string format = ".bin";
    // cv::Mat storageMat = cv::Mat::zeros(4500, 1, CV_32F);
    cv::Mat storageMat = cv::Mat::zeros(1, 300, CV_32F);
    //HRV_info HRVi;
    GetAllFormatFiles(folder, files, format);
    std::ofstream csv_file(folder + "ppg.csv");
    std::ofstream csv_file_ppgval(folder + "ppg_val.csv");
    std::ofstream csv_file_bre(folder + "bre.csv");
    std::ofstream csv_file_breval(folder + "bre_val.csv");
    std::ofstream csv_face(folder + "face_box.csv");
    csv_face << "frame,face_x,face_y,face_w,face_h\n";   // 写表头
    DataProcessor
            HRProcessor(FPS * 5);
    DataProcessor
            RRProcessor(FPS * 5);

    std::string roiFilename = "../rois.csv";
    std::string roi_chest_Filename = "../rois_chest.csv";
    std::string model_path = "/Users/yangxh/code/i20/i20test/face_detection_yunet_2023mar.onnx";
    YuNet face_detector(model_path); // 创建 YuNet 人脸检测对象

    for (int file_idx = 0; file_idx < files.size(); file_idx++)
    {
        bin2mat b2m(files.at(file_idx));

        b2m.set_frame(start_frame);
        csv_file << files.at(file_idx);
        csv_file_ppgval << files.at(file_idx);
        csv_file_bre << files.at(file_idx);
        csv_file_breval << files.at(file_idx);
        cv::Mat im = b2m.get_mat();
        cv::Mat bgr;
        std::cout<<"图像尺寸"<<im.size()<<std::endl;
        processImage(im, bgr, db, b2m);
        pulse pul(dst_size, FPS);
        breath bre(dst_size, FPS);

        int HR = 0; float Qua = 0;

        for (int frame_idx = start_frame; (frame_idx < b2m.frameCount) && (frame_idx <10000); frame_idx++)
        {
            im = b2m.get_mat();
            processImage(im, bgr, db, b2m);
            if (frame_idx % 10 == 0) {
                bool face_detected = false;
                for (int angle = 0; angle < 360; angle += 45) {
                    cv::Mat rotated_im = rotateImage(im, angle);
                    face_detector.setInputSize(rotated_im.size());
                    auto faces = face_detector.infer(rotated_im);

                    if (!faces.empty()) {

                        auto face_boxes = getFaceBoxes(faces);
                        if (!face_boxes.empty()) {
                            cv::Rect rotatedFaceBox = face_boxes[0];
                            currentFaceBox = mapBackToOriginal(rotatedFaceBox, angle, im.size());

                            face_detected = true;
                            break;
                        }
                    }
                }

                if (!face_detected) {
                    std::cerr << "No faces detected in frame " << frame_idx + 1 << ". Skipping this frame." << std::endl;
                    currentFaceBox = lastFaceBox;
                }
            } else {
                // 非检测帧，使用上次检测的框
                currentFaceBox = lastFaceBox;
            }


            if (currentFaceBox.empty())
            {
                currentFaceBox = lastFaceBox;
            }
            currentFaceBox = adjust_rect(lastFaceBox, currentFaceBox, im.size());
            csv_face << frame_idx << ','
                     << currentFaceBox.x << ','
                     << currentFaceBox.y << ','
                     << currentFaceBox.width << ','
                     << currentFaceBox.height << '\n';
           // cv::rectangle(im, currentFaceBox, cv::Scalar(0, 255, 0) );
            imshow("bgr",im);
            lastFaceBox = currentFaceBox;

            cv::Mat face_bgr = bgr(currentFaceBox);
            cv::resize(face_bgr, face_bgr, dst_size, cv::INTER_AREA);
            cv::Mat chestroi;
            cv::Point center = (currentFaceBox.tl() + currentFaceBox.br()) * 0.5;
            int width = currentFaceBox.width * 4 ;
            int height = currentFaceBox.height * 4;
            int x = center.x - width / 2;
            int y = center.y - height / 2;

            cv::Rect chest_roi(x, y, width, height);
//            chest_roi = chest_roi & cv::Rect(0, 0, bgr.cols, bgr.rows);
//            chestroi = bgr(chest_roi);
//
//            //单选胸框解除
//            //chestroi = bgr(rois_chest.at(file_idx));
//            cv::resize(chestroi, chestroi, dst_size, cv::INTER_AREA);
            // ---------- 2. 计算需要填充的边界 ----------
            int pad_left   = std::max(0, -chest_roi.x);
            int pad_top    = std::max(0, -chest_roi.y);
            int pad_right  = std::max(0, chest_roi.br().x - bgr.cols);
            int pad_bottom = std::max(0, chest_roi.br().y - bgr.rows);
            cv::Mat bgr_padded;
            cv::copyMakeBorder(bgr, bgr_padded,
                               pad_top, pad_bottom, pad_left, pad_right,
                               cv::BORDER_REPLICATE);   // 也可用 BORDER_REFLECT_101

            cv::Rect chest_roi_padded = chest_roi + cv::Point(pad_left, pad_top);
            cv::resize(bgr_padded(chest_roi_padded), chestroi, dst_size, 0, 0, cv::INTER_AREA);

            pulse_info pi = pul.detect(face_bgr, cv::Rect(0, 0, face_bgr.cols, face_bgr.rows), false);
            cv::Mat R_signal, G_signal, B_signal;
            cv::Mat rgb_trace = cv::Mat::zeros(10 * FPS, 3, CV_32FC1);
            breath_info bi = bre.detect(chestroi);
            /******平滑*********/
            RRProcessor.addDataPoint(bi.breath_rates.back());
            HRProcessor.addDataPoint(pi.heartrate);
            if (!bi.breath_signal.empty()){
                bpm_value = int(bi.breath_rates.back()) == 0 ? int(bi.breath_rates.back()) : int(RRProcessor.calculateAverage());
            }

            ppg_bpm_value = pi.heartrate == 0 ? pi.heartrate : HRProcessor.calculateAverage();

            /******平滑*********/
//                cv::Mat rates_img = showRates(pi.heartrate, bpm_value);
//                cv::imshow("Heart Rate and Breath Rate", rates_img);



            cv::Mat ppgPlot = drawSignalPlot(pi.ppg_signal,ppg_bpm_value,1200,200,300);
            cv::imshow("PPG Signal", ppgPlot);
            cv::Mat brePlot = drawSignalPlot(bi.breath_signal,bpm_value,1200,200,300);
            cv::imshow("BRE Signal", brePlot);



            if (!pul.bbox_topK_bgr.empty())
            {
                for (auto box : pul.bbox_topK_bgr)
                {
                    cv::rectangle(face_bgr, box, cv::Scalar::all(255));
                }
                for(auto box:bi.bbox_select)
                {
                    cv::rectangle(chestroi, box, cv::Scalar::all(255));
                }
            }
            cv::Mat enlarged_face_bgr, enlarged_chestroi;
            cv::resize(face_bgr, enlarged_face_bgr, cv::Size(face_bgr.cols * 4, face_bgr.rows * 4));
            cv::resize(chestroi, enlarged_chestroi, cv::Size(chestroi.cols * 4, chestroi.rows * 4));

            cv::Mat combined_image;
            hconcat(enlarged_face_bgr, enlarged_chestroi, combined_image);
            imshow("Combined Image", combined_image);


            if (pi.ppg_signal.size() == signal_len) {

                csv_file << "," << *(pi.ppg_signal.begin());
                csv_file_ppgval << "," <<pi.heartrate;
                //csv_file_ppgval << "," <<ppg_bpm_value;

            }

            if (bi.breath_signal.size() >= 225) {

                csv_file_bre << "," << *(bi.breath_signal.begin());
                csv_file_breval << "," << bi.breath_rates.back();
                //csv_file_breval << "," << bpm_value;

            }


            std::cout << "file: " << file_idx + 1 << "/" << files.size() << " frame: " << frame_idx + 1 << "/" << b2m.frameCount  << " size = " << pul.bbox.size() << std::endl;
            cv::waitKey(1);
            // cv::waitKey(1000/FPS);
        }
        csv_file << "\n";
        csv_file_ppgval << "\n";
        csv_file_bre << "\n";
        csv_file_breval << "\n";

    }
    csv_file.close();
    csv_file_ppgval.close();
    csv_file_bre.close();
    csv_file_breval.close();
    csv_face.close();

    return 0;
}
