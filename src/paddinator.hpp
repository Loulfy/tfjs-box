#ifndef PADDINATOR_H
#define PADDINATOR_H

#include<iostream>

#include <nan.h>
#include <opencv2/opencv.hpp>

class Paddinator : public Nan::ObjectWrap
{
  public:
    static void Init(v8::Local<v8::Object> exports);
    
  private:
    explicit Paddinator(float* data, cv::Size shape, cv::Size target);
    ~Paddinator();
    
    static void New(const Nan::FunctionCallbackInfo<v8::Value>& info);
    static void Process(const Nan::FunctionCallbackInfo<v8::Value>& info);
    static void Restore(const Nan::FunctionCallbackInfo<v8::Value>& info);
    static Nan::Persistent<v8::Function> constructor;
    
    cv::Mat& process();
    cv::Mat& restore(float* data);
    size_t length() const;
    
    cv::Mat m_image;
    cv::Mat m_target;
};

#endif
