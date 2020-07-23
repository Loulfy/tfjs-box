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
  if(!info[2]->IsArray()) return Nan::ThrowTypeError("Argument 2 must be [h, s]");
  TypedArrayContents<float> data(info[0]);
  v8::Local<v8::Context> context = info.GetIsolate()->GetCurrentContext();
  
  double w = 0, h = 0;
  v8::Local<v8::Array> shape = v8::Local<v8::Array>::Cast(info[1]);
  if(Has(shape, 0).FromJust() && Has(shape, 1).FromJust())
  {
    h = Nan::Get(shape, 0).ToLocalChecked()->NumberValue(context).FromJust();
    w = Nan::Get(shape, 1).ToLocalChecked()->NumberValue(context).FromJust();
  }
  
  size_t i;
  std::vector<float> hue;
  v8::Local<v8::Array> color = v8::Local<v8::Array>::Cast(info[2]);
  for(i = 0; i < 2; ++i) if(Has(color, i).FromJust()) hue.push_back(Nan::Get(color, i).ToLocalChecked()->NumberValue(context).FromJust());
  
  cv::Mat img(h, w, CV_32FC3, (*data));
  
  cv::cvtColor(img, img, cv::COLOR_RGB2HSV);
  
  cv::Mat hsv[3];
  cv::split(img, hsv);
  
  for(i = 0; i < hue.size(); ++i) hsv[i] = cv::Scalar::all(hue[i]);
  
  cv::merge(hsv, 3, img);

  cv::cvtColor(img, img, cv::COLOR_HSV2RGB);

  info.GetReturnValue().SetUndefined();
}

NAN_MODULE_INIT(Init) {
  NAN_EXPORT(target, adjust_hue);
  Paddinator::Init(target);
}

NODE_MODULE(typedarrays, Init)
