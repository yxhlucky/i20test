#include "main.h"


int main()
{
    //std::string folder = "/Users/yangxh/Data/i20_Data/ppg/";
    //std::string folder = "/Volumes/637data_expansion/yxh/i20/i20_data/ppg_bre/syy_ppg_bre/";
    std::string folder = "/Users/yangxh/Data/i20_Data/sean/";
    //std::string folder = "/Users/yangxh/Data/i20_Data/0617/wwj/";
    //std::string folder = "/Users/yangxh/Data/i20_Data/0618/yxh_20/";
    //std::string folder = "/Users/yangxh/Data/i20_Data/0719/";
    //std::string folder = "/Users/yangxh/Data/i20_Data/lily_baby_0624/10min_1.1m/";
    //std::string folder = "/Users/yangxh/Data/i20_Data/0624/yxh_nir/";
    //开关  false 开启  true 关闭
    bool loadFromFile = false;
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
    DataProcessor
    HRProcessor(FPS * 5);
    DataProcessor
    RRProcessor(FPS * 5);

    std::string roiFilename = "../rois.csv";
    std::string roi_chest_Filename = "../rois_chest.csv";
    if (loadFromFile) {
        loadROIs(rois, roiFilename);
    }
    else {
        for (int file_idx = 0; file_idx < files.size(); file_idx++) {
            bin2mat b2m(files.at(file_idx));
            if (b2m.frameCount > 50) b2m.set_frame(1000);

            cv::Mat im = b2m.get_mat();
            cv::Mat bgr;
            processImage(im, bgr, db, b2m);

            b2m.close();

            // cv::imshow("Draw Rectangle", enlarged_bgr);
            rois.push_back(rd.drawRectangle(bgr));
            //单选胸框解除
            //rois_chest.push_back(rd.drawRectangle(bgr));
        }
        saveROIs(rois, roiFilename);
        //单选胸框解除
        //saveROIs(rois_chest, roiFilename);
    }

    if (rois.size() == files.size())
    {

        for (int file_idx = 0; file_idx < files.size(); file_idx++)
        {
            bin2mat b2m(files.at(file_idx));
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

            for (int frame_idx = 0; (frame_idx < b2m.frameCount) && (frame_idx <4600); frame_idx++)
            {

                im = b2m.get_mat();
                processImage(im, bgr, db, b2m);
                cv::Mat face_bgr = bgr(rois.at(file_idx));
                cv::resize(face_bgr, face_bgr, dst_size, cv::INTER_AREA);
                cv::Mat chestroi;

                cv::Rect roi = cv::Rect(rois.at(file_idx).x,
                                        rois.at(file_idx).y/1.1 + rois.at(file_idx).height,
                                        rois.at(file_idx).width , rois.at(file_idx).height );
                cv::Point center = (rois.at(file_idx).tl() + rois.at(file_idx).br()) * 0.5;
                int width = rois.at(file_idx).width * 4 ;
                int height = rois.at(file_idx).height * 4;
                int x = center.x - width / 2;
                int y = center.y - height / 2;

                cv::Rect chest_roi(x, y, width, height);

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

//                chest_roi = chest_roi & cv::Rect(0, 0, bgr.cols, bgr.rows);
//                chestroi = bgr(chest_roi);
//
//                //单选胸框解除
//                //chestroi = bgr(rois_chest.at(file_idx));
//                cv::resize(chestroi, chestroi, dst_size, cv::INTER_AREA);
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


                // 使用
//                cv::Mat ppg_mat(150, 1, CV_32F);
//                if(!pi.ppg_signal.empty()){
//                    update_signal_trace_cols(pi.ppg_signal.back(), storageMat, storageMat.cols);
//
//                }

//                if (storageMat.cols >= WIN_LEN_HRV * FPS) {
//                    if (frameCount_HRV % (FPS * 60) == 0) {
//                        std::cout<<"12321"<<std::endl;
//                        int length = (int) storageMat.cols;
//                        cv::Mat trace = storageMat.row(1).colRange(length - WIN_LEN_HRV * FPS,
//                                                                 length).clone();
//                        HRVi = HRV_processor.getHRV(trace);
//                    }
//                    frameCount_HRV++;
//                    frameCount_HRV == (FPS * 60) ? frameCount_HRV = 0 : frameCount_HRV;
//                }

//                std::cout << "PPG Signal Mat: [";
//                for (int i = 0; i < storageMat.rows; i++) {
//                    std::cout << storageMat.at<float>(i, 0);
//                    if (i < storageMat.rows - 1) std::cout << ", ";
//                }
//                std::cout << "]" << std::endl;



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

    }
    else
        return -1;

    csv_file.close();
    csv_file_ppgval.close();
    csv_file_bre.close();
    csv_file_breval.close();


    return 0;
}
