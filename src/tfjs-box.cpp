#include <nan.h>
#include <stdint.h>

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "paddinator.hpp"

using namespace Nan;

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void print(cv::Mat& mat)
{
  std::cout << mat.size() << " = " << type2str(mat.type()) << std::endl;
}

NAN_METHOD(adjust_hue)
{
  if(!info[0]->IsFloat32Array()) return Nan::ThrowTypeError("Argument 0 must be Float32Array");
  if(!info[1]->IsArray()) return Nan::ThrowTypeError("Argument 1 must be [height, width]");
  if(!info[2]->IsArray()) return Nan::ThrowTypeError("Argument 2 must be [h, s, v]");
  TypedArrayContents<float> data(info[0]);
  v8::Local<v8::Context> context = info.GetIsolate()->GetCurrentContext();
  
  double wt = 0, ht = 0;
  v8::Local<v8::Array> shape = v8::Local<v8::Array>::Cast(info[1]);
  if(Has(shape, 0).FromJust() && Has(shape, 1).FromJust())
  {
    ht = Nan::Get(shape, 0).ToLocalChecked()->NumberValue(context).FromJust();
    wt = Nan::Get(shape, 1).ToLocalChecked()->NumberValue(context).FromJust();
  }
  
  size_t i;
  std::vector<float> target;
  v8::Local<v8::Array> color = v8::Local<v8::Array>::Cast(info[2]);
  for(i = 0; i < 3; ++i) if(Has(color, i).FromJust()) target.push_back(Nan::Get(color, i).ToLocalChecked()->NumberValue(context).FromJust());
  
  if(target.empty())
  {
    info.GetReturnValue().SetUndefined();
    return;
  }
  
  cv::Mat img(ht, wt, CV_32FC3, (*data));
  
  cv::cvtColor(img, img, cv::COLOR_RGB2HSV);
  
  cv::Mat labels, centers;
  cv::Mat dom = img.reshape(1, img.total());
  cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10, 1.0);
  cv::kmeans(dom, 1, labels, criteria, 10, cv::KMEANS_RANDOM_CENTERS, centers);
  centers = centers.reshape(3,centers.rows);
  float* vibrant = centers.ptr<float>();
  
  float h, s, v;
  for (int j = 0; j < img.rows; ++j)
  {
    for (int i = 0; i < img.cols; ++i)
    {      
      h = img.at<cv::Vec3f>(j, i)[0] + target[0] - vibrant[0];
      if(h < 0) h+= 360;
      if(h >= 360) h-= 360;
      img.at<cv::Vec3f>(j, i)[0] = h;
      
      if(target.size() > 1)
      {
        s = img.at<cv::Vec3f>(j, i)[1];
        if(s < vibrant[1]) img.at<cv::Vec3f>(j, i)[1] = target[1]/vibrant[1] * s;
        else img.at<cv::Vec3f>(j, i)[1] = (1 - target[1])/(1 - vibrant[1]) * s + (target[1] - vibrant[1])/(1 - vibrant[1]);
      }
      
      if(target.size() > 2)
      {
        v = img.at<cv::Vec3f>(j, i)[2];
        if(v < vibrant[2]) img.at<cv::Vec3f>(j, i)[2] = target[2]/vibrant[2] * v;
        else img.at<cv::Vec3f>(j, i)[2] = (255 - target[2])/(255 - vibrant[2]) * v + 255*(target[2] - vibrant[2])/(255 - vibrant[2]);
      }
    }
  }
  
  cv::cvtColor(img, img, cv::COLOR_HSV2RGB);

  info.GetReturnValue().SetUndefined();
}

NAN_MODULE_INIT(Init) {
  NAN_EXPORT(target, adjust_hue);
  Paddinator::Init(target);
}

NODE_MODULE(typedarrays, Init)
