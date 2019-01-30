#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <torch/script.h> // One-stop header.
#include <stdio.h>

#include <iostream>
#include <memory>

std::shared_ptr<torch::jit::script::Module> MODULE;

using namespace cv;

Mat binarizeImg(Mat g_frame)
{
  int rows = g_frame.rows;
  int cols = g_frame.cols;
  threshold(g_frame, g_frame, 170, 255, 0); //For the thresholding part
  Rect roi;
  roi.x = cols/4;
  roi.width = cols/2;
  roi.y = rows/4 - 50;
  roi.height = rows/2 + 100;

  g_frame = g_frame(roi);

  return g_frame;
}

Mat cleanUpNoise(Mat noisy_img)
{
  Mat img;
  Mat element = getStructuringElement(MORPH_RECT, Size(5, 5)); //Maybe make this 3, 3
  erode(noisy_img, img, element);
  Mat element2 = getStructuringElement(MORPH_RECT, Size(10, 10)); //Maybe make this 3, 3
  dilate(img, img, element2);

  return img;
}

std::vector<Point2f> findCentroids(Mat diff)
{
  Mat canny_out;
  Canny(diff, canny_out, 100, 200, 3); //May need to change the middle two values
  std::vector<std::vector<Point>> contours;
  findContours(canny_out, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

  std::vector<Moments> mu(contours.size());
  std::vector<Point2f> mc(contours.size());
  for(int i(0); i< contours.size(); i++)
  {
    mu[i] = moments(contours[i]);
    mc[i] = Point2f(static_cast<float>(mu[i].m10/(mu[i].m00+1e-5)),
                    static_cast<float>(mu[i].m01/(mu[i].m00+1e-5)));
  }

  return mc;
}

Mat addTextToImage(Mat gray_img, int num_loops)
{
  std::string label;
  if(num_loops == 1)
    label = "1 Loop";
  else if(num_loops == 2)
    label = "2 Loops";
  else if(num_loops == 3)
    label = "3 Loops";
  else
    label = "";

  // Mat img = gray_img.copy();
  // cvtColor(gray_img, img, cv::COLOR_GRAY2BGR);
  putText(gray_img, label.c_str(), Point(200, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255));
  //parameters: output, label, origin, font type, font scale, color
  // Mat color_img;



  return gray_img;
}

int get_class_index(cv::Mat inputImage)
{
  std::vector<int64_t> sizes = {1, 1, inputImage.rows, inputImage.cols};
  at::TensorOptions options(at::ScalarType::Byte);
  at::Tensor tensor_image = torch::from_blob(inputImage.data, at::IntList(sizes), options);
  tensor_image = tensor_image.toType(at::kFloat);

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(tensor_image.cuda());
  at::Tensor result = MODULE->forward(inputs).toTensor();

  at::Tensor max_index = result.argmax();
  auto identity = torch::ones({}, options).toType(at::kLong).cuda();

  int r = -1;
  if(max_index.equal(identity))
    r = 2;
  else if(max_index.equal(identity*2))
    r = 3;
  else
    r = 1;
  return r;
}

int main()
{

  // Loading the Nueral Net Model
  MODULE = torch::jit::load("../model.pt");

  assert(MODULE != nullptr);
  std::cout << "Model loaded. Ready to Go!\n";

  VideoCapture video(1);
  if(!video.isOpened())
  {
      std::cout << "Failed to open camera\n";
      return -1;
  }

  Mat frame, g_frame, img;
  int key, num_loops(1);

  cv::Mat display_image;
  video >> display_image;

  while(true)
  {
    video >> frame;
    cvtColor(frame, g_frame, COLOR_BGR2GRAY);

    g_frame = binarizeImg(g_frame);
    g_frame = cleanUpNoise(g_frame);



    std::vector<Point2f> centroids = findCentroids(g_frame); //Note: There should only be 1 (for the pretzel)
    //Locate where centroid of pretzel is >100 & < 300

    if(centroids.size() > 0)
    {
      if(centroids[0].y > 100 && centroids[0].y < 150)
      {
        // int size = 250;
        // Rect roi;
        // roi.x = centroids[0].x - size/2;
        // roi.width = size;
        // roi.y = centroids[0].y - size/2;
        // roi.height = size;

        cv::Mat toCNN;
        // toCNN = g_frame(roi);
        g_frame.copyTo(display_image);
        //Pass image into DNN. Return is an int showing the number of loops
        cv::resize(g_frame, toCNN, cv::Size(45,45));
        toCNN = toCNN / 255.0;
        num_loops = get_class_index(toCNN);
      }
      else
      {
        num_loops = 0;
      }
    }
    else
    {
      num_loops = 0;
    }

    // std::cout << num_loops << std::endl;
    display_image = addTextToImage(display_image, num_loops);

    imshow("Color", frame);
    imshow("Live", g_frame);
    imshow("Still", display_image);
    key = waitKey(10);
    if(key == (int)('q'))
      break;
  }

  return 0;
}
