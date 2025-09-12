#include <opencv2/opencv.hpp>
#include <iostream>

class RectangleDrawer {
public:
    RectangleDrawer() : drawing(false), finished(false) {}

    cv::Rect drawRectangle(cv::Mat image) {
        finished = false;
        if (image.empty()) {
            std::cerr << "Image is empty!" << std::endl;
            return cv::Rect();
        }

        cv::namedWindow("Draw Rectangle", cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback("Draw Rectangle", onMouse, this);

        this->image = image;
        this->tempImage = image.clone();

        while (!finished) {
            cv::imshow("Draw Rectangle", tempImage);
            char key = (char)cv::waitKey(1);

            if (key == 27) { // ESC key
                break;
            }
        }

        cv::destroyWindow("Draw Rectangle");
        return rectangle;
    }

private:
    static void onMouse(int event, int x, int y, int flags, void* userdata) {
        RectangleDrawer* drawer = reinterpret_cast<RectangleDrawer*>(userdata);
        drawer->handleMouseEvent(event, x, y, flags);
    }

    void handleMouseEvent(int event, int x, int y, int flags) {
        if (event == cv::EVENT_LBUTTONDOWN) {
            startPoint = cv::Point(x, y);
            drawing = true;
        }
        else if (event == cv::EVENT_MOUSEMOVE && drawing) {
            tempImage = image.clone();
            cv::rectangle(tempImage, startPoint, cv::Point(x, y), cv::Scalar(255, 255, 255), 2);
        }
        else if (event == cv::EVENT_LBUTTONUP && drawing) {
            endPoint = cv::Point(x, y);
            drawing = false;
            rectangle = cv::Rect(startPoint, endPoint);
            cv::rectangle(image, startPoint, endPoint, cv::Scalar(255, 255, 255), 2);
            finished = true; 
        }
    }

    cv::Mat image;
    cv::Mat tempImage;
    cv::Point startPoint;
    cv::Point endPoint;
    cv::Rect rectangle;
    bool drawing;
    bool finished; 
};
