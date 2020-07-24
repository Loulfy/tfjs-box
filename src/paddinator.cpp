#include "paddinator.hpp"

Nan::Persistent<v8::Function> Paddinator::constructor;

Paddinator::Paddinator(float* data, cv::Size shape, cv::Size target) : m_image(shape, CV_32FC3), m_target(target, CV_32FC3)
{
  float* contents = m_image.ptr<float>();
  for(int i = 0; i < shape.area()*3; ++i) contents[i] = data[i];
}

Paddinator::~Paddinator() {}

void Paddinator::Init(v8::Local<v8::Object> exports)
{
  v8::Local<v8::Context> context = exports->CreationContext();
  Nan::HandleScope scope;
  
  // Prepare constructor template
  v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
  tpl->SetClassName(Nan::New("Paddinator").ToLocalChecked());
  tpl->InstanceTemplate()->SetInternalFieldCount(1);
  
  // Prototype
  Nan::SetPrototypeMethod(tpl, "process", Process);
  Nan::SetPrototypeMethod(tpl, "restore", Restore);
  
  constructor.Reset(tpl->GetFunction(context).ToLocalChecked());
  auto err = exports->Set(context, Nan::New("Paddinator").ToLocalChecked(), tpl->GetFunction(context).ToLocalChecked());
  if(!err.FromJust()) Nan::ThrowError("Paddinator Init failed");
}

void Paddinator::New(const Nan::FunctionCallbackInfo<v8::Value>& info)
{
  v8::Local<v8::Context> context = info.GetIsolate()->GetCurrentContext();
  if (info.IsConstructCall())
  {
    if(!info[0]->IsFloat32Array()) return Nan::ThrowTypeError("Argument 0 must be Float32Array");
    if(!info[1]->IsArray()) return Nan::ThrowTypeError("Argument 1 must be [width, height]");
    if(!info[2]->IsArray()) return Nan::ThrowTypeError("Argument 2 must be [new width, new height]");
    
    Nan::TypedArrayContents<float> data(info[0]);
    
    double w = 0, h = 0;
    v8::Local<v8::Array> array;
    
    array = v8::Local<v8::Array>::Cast(info[1]);
    if(Nan::Has(array, 0).FromJust() && Nan::Has(array, 1).FromJust())
    {
      h = Nan::Get(array, 0).ToLocalChecked()->NumberValue(context).FromJust();
      w = Nan::Get(array, 1).ToLocalChecked()->NumberValue(context).FromJust();
    }
    
    cv::Size shape(w, h);
    
    array = v8::Local<v8::Array>::Cast(info[2]);
    if(Nan::Has(array, 0).FromJust() && Nan::Has(array, 1).FromJust())
    {
      h = Nan::Get(array, 0).ToLocalChecked()->NumberValue(context).FromJust();
      w = Nan::Get(array, 1).ToLocalChecked()->NumberValue(context).FromJust();
    }
    
    cv::Size target(w, h);
    
    Paddinator* obj = new Paddinator(*data, shape, target);
    obj->Wrap(info.This());
    info.GetReturnValue().Set(info.This());
  }
  else
  {
    const int argc = 2;
    v8::Local<v8::Value> argv[argc] = {info[0], info[1]};
    v8::Local<v8::Function> cons = Nan::New<v8::Function>(constructor);
    info.GetReturnValue().Set(cons->NewInstance(context, argc, argv).ToLocalChecked());
  }
}

void Paddinator::Process(const Nan::FunctionCallbackInfo<v8::Value>& info)
{
  v8::Local<v8::Context> context = info.GetIsolate()->GetCurrentContext();
  auto pad = ObjectWrap::Unwrap<Paddinator>(info.Holder());
  auto img = pad->process();
  
  int length = img.size().area()*3;
  v8::Local<v8::ArrayBuffer> buf = v8::ArrayBuffer::New(info.GetIsolate(), length*sizeof(float));
  std::memcpy(buf->GetContents().Data(), img.data, length*sizeof(float));
  v8::Local<v8::Float32Array> array = v8::Float32Array::New(buf, 0, length);
  
  v8::Local<v8::Object> obj = Nan::New<v8::Object>();
  v8::Local<v8::Array> shape = Nan::New<v8::Array>(3);
  Nan::Set(shape, 0, Nan::New<v8::Number>(img.size().height));
  Nan::Set(shape, 1, Nan::New<v8::Number>(img.size().width));
  Nan::Set(shape, 2, Nan::New<v8::Number>(3));
  
  auto a = obj->Set(context, Nan::New("shape").ToLocalChecked(), shape);
  if(!a.FromJust()) Nan::ThrowError("Paddinator Process failed (shape)");
  auto b = obj->Set(context, Nan::New("array").ToLocalChecked(), array);
  if(!b.FromJust()) Nan::ThrowError("Paddinator Process failed (array)");
  info.GetReturnValue().Set(obj);
}

cv::Mat& Paddinator::process()
{
  auto source = m_image.size();
  auto target = m_target.size();
  
  double ratio = std::max((double)source.width / target.width, (double)source.height / target.height);
  
  cv::Size resized(std::floor(source.width/ratio), std::floor(source.height/ratio));
  
  cv::resize(m_image, m_image, resized);
  
  int delta_w = target.width - resized.width;
  int delta_h = target.height - resized.height;
  
  cv::copyMakeBorder(m_image, m_target, 0, delta_h, 0, delta_w, cv::BORDER_CONSTANT, cv::Scalar::all(0));
  return m_target;
}

void Paddinator::Restore(const Nan::FunctionCallbackInfo<v8::Value>& info)
{
  v8::Local<v8::Context> context = info.GetIsolate()->GetCurrentContext();
  auto pad = ObjectWrap::Unwrap<Paddinator>(info.Holder());
  
  if(!info[0]->IsFloat32Array())
  {
    std::string msg = "Argument must be Float32Array("+std::to_string(pad->length())+")";
    return Nan::ThrowTypeError(msg.c_str());
  }
  
  Nan::TypedArrayContents<float> data(info[0]);
  
  auto img = pad->restore(*data);
  
  int length = img.size().area()*3;
  v8::Local<v8::ArrayBuffer> buf = v8::ArrayBuffer::New(info.GetIsolate(), length*sizeof(float));
  
  //std::memcpy(buf->GetContents().Data(), img.data, length*sizeof(float));
  
  int w = img.size().width;
  int h = img.size().height;
  for(int i = 0; i < h; ++i)
  {
    cv::Mat row = img.row(i);
    std::memcpy((uchar*)buf->GetContents().Data()+i*w*3*sizeof(float), row.data, w*3*sizeof(float));
  }
  
  v8::Local<v8::Float32Array> array = v8::Float32Array::New(buf, 0, length);
  
  v8::Local<v8::Object> obj = Nan::New<v8::Object>();
  v8::Local<v8::Array> shape = Nan::New<v8::Array>(3);
  Nan::Set(shape, 0, Nan::New<v8::Number>(img.size().height));
  Nan::Set(shape, 1, Nan::New<v8::Number>(img.size().width));
  Nan::Set(shape, 2, Nan::New<v8::Number>(3));
  
  auto a = obj->Set(context, Nan::New("shape").ToLocalChecked(), shape);
  if(!a.FromJust()) Nan::ThrowError("Paddinator Process failed (shape)");
  auto b = obj->Set(context, Nan::New("array").ToLocalChecked(), array);
  if(!b.FromJust()) Nan::ThrowError("Paddinator Process failed (array)");
  info.GetReturnValue().Set(obj);
}

cv::Mat& Paddinator::restore(float* data)
{  
  float* contents = m_target.ptr<float>();
  for(int i = 0; i < m_target.size().area()*3; ++i) contents[i] = data[i];
  
  auto size = m_image.size();
  m_image = cv::Mat(m_target, cv::Rect{0, 0, size.width, size.height});
  return m_image;
}

size_t Paddinator::length() const
{
  return m_target.size().area()*3;
}
