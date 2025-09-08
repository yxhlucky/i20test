#include "eye.h"

eye::eye(std::string classifier_name, std::string model_path, int fps)
{
    hog = cv::HOGDescriptor(cv::Size(20, 20), cv::Size(10, 10), cv::Size(5, 5), cv::Size(10, 10), 9, 1, -1.0, cv::HOGDescriptor::L2Hys, 0.2, true, 64, true);
    //hog = cv::HOGDescriptor(cv::Size(20, 20), cv::Size(10, 10), cv::Size(5, 5), cv::Size(10, 10), 9);

    param.classifier_name = classifier_name;
    param.eyebuf_len = fps * 5; // 5 sec
    param.actbuf_len = fps * 30; // 30 sec

    if (classifier_name == "svm")
    {
        svm = cv::ml::SVM::load(model_path);
    }

    if (classifier_name == "mlp")
    {
        mlp = cv::ml::ANN_MLP::load(model_path);
    }
}

sleep_info eye::detect(cv::Mat& rgb, cv::Rect faceroi, cv::Point2i lefteye, cv::Point2i righteye, float activity)
{
    // double t1 = (double)cv::getTickCount();
   
    cv::Mat feat = extract_hogfeat2(rgb, faceroi, lefteye, righteye);
   
    float score = classify_hogfeat(feat);
    
    // time average
    actbuf.push_back(activity);
    if (actbuf.size() > param.actbuf_len)
    {
        actbuf.pop_front();
    }

    float activity_avg = std::accumulate(actbuf.begin(), actbuf.end(), 0.0) / (1e-6 + actbuf.size());

    int state = (score > 0 || activity_avg > 3) ? 1 : 0;
    sleep_info si = { state, score };

    // double t2 = (double)cv::getTickCount();
    // std::cout << "Speed :" << (t2 - t1) / cv::getTickFrequency() << std::endl;

    return si;
}

cv::Mat eye::gen_hogfeat(cv::Mat rgb, cv::Mat face)
{
    if (face.empty())
    {
        return cv::Mat();
    }
    else
    {
        cv::Mat gray;
        cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

        int idx = 0;
        int block_size = (int)(face.at<float>(idx, 2) / 5);

        cv::Point2i leye_xy(int(face.at<float>(idx, 4)), int(face.at<float>(idx, 5)));
        cv::Rect leye_roi(leye_xy.x - block_size / 2, leye_xy.y - block_size / 2, block_size, block_size);

        cv::Point2i reye_xy(int(face.at<float>(idx, 6)), int(face.at<float>(idx, 7)));
        cv::Rect reye_roi(reye_xy.x - block_size / 2, reye_xy.y - block_size / 2, block_size, block_size);

        leye_roi = leye_roi & cv::Rect(0, 0, gray.cols, gray.rows);
        reye_roi = reye_roi & cv::Rect(0, 0, gray.cols, gray.rows);

        if (leye_roi.area() > 0 && reye_roi.area() > 0)
        {
            cv::Mat leye = gray(leye_roi).clone();
            cv::Mat reye = gray(reye_roi).clone();

            int roi_size = 20;
            cv::resize(leye, leye, cv::Size(roi_size, roi_size), 0, 0, cv::INTER_AREA);
            cv::resize(reye, reye, cv::Size(roi_size, roi_size), 0, 0, cv::INTER_AREA);

            std::vector<float> leye_feat, reye_feat;
            hog.compute(leye, leye_feat);
            hog.compute(reye, reye_feat);

            cv::Mat leye_feat_mat = cv::Mat(1, leye_feat.size(), CV_32FC1, leye_feat.data());
            cv::Mat reye_feat_mat = cv::Mat(1, reye_feat.size(), CV_32FC1, reye_feat.data());

            cv::Mat feat;
            cv::vconcat(leye_feat_mat, reye_feat_mat, feat);

            return feat;
        }
    }
}

cv::Mat eye::extract_hogfeat(cv::Mat& rgb, cv::Rect faceroi, cv::Point2i lefteye, cv::Point2i righteye)
{
    if (faceroi.area() == 0)
    {
        return cv::Mat();
    }
    else
    {
        cv::Mat gray;
        cv::cvtColor(rgb, gray, cv::COLOR_BGR2GRAY);

        int block_size = (int)(faceroi.width / 5);
        int scan_dist = 3, scan_step = 1;
        int roi_size = 20;

        cv::Mat feat;
        std::vector<float> feat_vec;

        for (int x = -scan_dist; x <= scan_dist; x = x + scan_step)
        {
            for (int y = -scan_dist; y <= scan_dist; y = y + scan_step)
            {
                cv::Rect leye_roi(lefteye.x + x - block_size / 2, lefteye.y + y - block_size / 2, block_size, block_size);
                leye_roi = leye_roi & cv::Rect(0, 0, gray.cols, gray.rows);
                if (leye_roi.area() > 0)
                {
                    cv::Mat leye = gray(leye_roi);
                    cv::resize(leye, leye, cv::Size(roi_size, roi_size)); // , 0, 0, cv::INTER_AREA

                    std::vector<cv::Point> locations;
                    hog.compute(leye, feat_vec);
                    feat.push_back(cv::Mat(1, feat_vec.size(), CV_32FC1, feat_vec.data()));
                }

                cv::Rect reye_roi(righteye.x + x - block_size / 2, righteye.y + y - block_size / 2, block_size, block_size);
                reye_roi = reye_roi & cv::Rect(0, 0, gray.cols, gray.rows);
                if (reye_roi.area() > 0)
                {
                    cv::Mat reye = gray(reye_roi);
                    cv::resize(reye, reye, cv::Size(roi_size, roi_size)); // , 0, 0, cv::INTER_AREA

                    hog.compute(reye, feat_vec);
                    feat.push_back(cv::Mat(1, feat_vec.size(), CV_32FC1, feat_vec.data()));
                }
            }
        }
        return feat;
    }
}


cv::Mat eye::extract_hogfeat2(cv::Mat& rgb, cv::Rect faceroi, cv::Point2i lefteye, cv::Point2i righteye)
{
    if (faceroi.area() == 0)
    {
        return cv::Mat();
    }
    else
    {
       
        cv::Rect roi = faceroi & cv::Rect(0, 0, rgb.cols, rgb.rows);
        if (roi.area() == 0)
            return cv::Mat();
       
        cv::Mat face = rgb(roi);
        // cv::cvtColor(face, face, cv::COLOR_BGR2GRAY);

        float ratio = (float)100.0 / face.cols;
        cv::Mat face_rz;
        cv::resize(face, face_rz, cv::Size(), ratio, ratio, cv::INTER_AREA);// , ratio, ratio);
        //cv::resize(face, face_rz, cv::Size(100, 120), 0 , 0 , cv::INTER_AREA);// , ratio, ratio);

        //cv::Point leye_cent = (lefteye - roi.tl()) * ratio;
        //cv::Point reye_cent = (righteye - roi.tl()) * ratio;

        cv::Point leye_cent((lefteye.x - roi.x) * ((float)face_rz.cols / face.cols), (lefteye.y - roi.y) * ((float)face_rz.rows / face.rows));
        cv::Point reye_cent((righteye.x - roi.x) * ((float)face_rz.cols / face.cols), (righteye.y - roi.y) * ((float)face_rz.rows / face.rows));

        int win = 20;
        int scan_dist = 3, scan_step = 1;
      
        std::vector<cv::Point> pts;
        for (int x = -scan_dist; x <= scan_dist; x = x + scan_step)
        {
            for (int y = -scan_dist; y <= scan_dist; y = y + scan_step)
            {
                cv::Point2i leye_pt = leye_cent + cv::Point(x, y) - cv::Point(win / 2, win / 2);
                cv::Point2i reye_pt = reye_cent + cv::Point(x, y) - cv::Point(win / 2, win / 2);

                if (leye_pt.x > 0 && leye_pt.y > 0 && leye_pt.x + win < face_rz.cols && leye_pt.y + win < face_rz.rows)
                {
                    pts.push_back(leye_pt);
                }

                if (reye_pt.x > 0 && reye_pt.y > 0 && reye_pt.x + win < face_rz.cols && reye_pt.y + win < face_rz.rows)
                {
                    pts.push_back(reye_pt);
                }
            }
        }
        
        //cv::Mat feats_mat = cv::Mat(pts.size(), feats.size() / pts.size(), CV_32FC1, feats.data());

        std::vector<float> feats;
        hog.compute(face_rz, feats, cv::Size(), cv::Size(), pts);
        
        cv::Mat feats_mat;
        if (pts.size() > 0)
        {
            feats_mat = cv::Mat::zeros(pts.size(), feats.size() / pts.size(), CV_32FC1);
            memcpy(feats_mat.data, feats.data(), feats.size() * sizeof(float));
        }
        
       
        //std::cout << pts.size() << "   " << feats_mat.size() << std::endl;
        /*
        //std::cout << feats_mat << std::endl;
        cv::circle(face_rz, leye_cent, 2, cv::Scalar(255, 0, 0));
        cv::circle(face_rz, reye_cent, 2, cv::Scalar(255, 0, 0));

        for (int i = 0; i < pts.size(); i++)
        {
            cv::Rect roi(pts.at(i).x, pts.at(i).y, 20, 20);
            cv::rectangle(face_rz, roi, cv::Scalar(255,0,0), 1);
            //cv::circle(face, pts.at(i), 1, cv::Scalar(0, 0, 255));
            //cv::circle(face, pts.at(i), 1, cv::Scalar(0, 0, 255));
        }
        cv::imshow("face", face_rz);
        */
        return feats_mat;
    }
}

float eye::classify_hogfeat(cv::Mat feat)
{
    if (!feat.empty())
    {
        float flag;

        if (param.classifier_name == "svm")
        {
            cv::Mat label;
            svm->predict(feat, label);
            flag = cv::mean(label).val[0];
        }

        if (param.classifier_name == "mlp")
        {
            cv::Mat label;
            mlp->predict(feat, label);

            // add confidence based selection
           // cv::Mat conf;
           // cv::reduce(label, conf, 0, cv::REDUCE_AVG);

            //cv::Mat mask;
            //cv::compare(label, 0.8, mask, cv::CMP_GT);
            //cv::reduce(mask, conf, 0, cv::REDUCE_SUM);
            // std::cout << "Conf = " << conf.at<float>(0, 0) << " === " << cv::mean(label.col(0)).val[0]  << std::endl;

            //flag = conf.at<float>(0,0) > conf.at<float>(0, 1) ? 1 : -1;

            flag = cv::mean(label.col(0)).val[0] > cv::mean(label.col(1)).val[0] ? 1 : -1;
        }

        // time average
        eyebuf.push_back(flag);
        if (eyebuf.size() > param.eyebuf_len)
        {
            eyebuf.pop_front();
        }
    }


    float score = std::accumulate(eyebuf.begin(), eyebuf.end(), 0.0) / (1e-6 + eyebuf.size());

    //std::cout << buf.size() << " " << score << std::endl;
    return score;
};

