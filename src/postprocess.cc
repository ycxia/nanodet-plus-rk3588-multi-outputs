// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <rknn_api.h>

#include <set>
#include <vector>
#include <math.h>
#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

static char* labels[OBJ_CLASS_NUM];

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

char* readLine(FILE* fp, char* buffer, int* len)
{
  int    ch;
  int    i        = 0;
  size_t buff_len = 0;

  buffer = (char*)malloc(buff_len + 1);
  if (!buffer)
    return NULL; // Out of memory

  while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
    buff_len++;
    void* tmp = realloc(buffer, buff_len + 1);
    if (tmp == NULL) {
      free(buffer);
      return NULL; // Out of memory
    }
    buffer = (char*)tmp;

    buffer[i] = (char)ch;
    i++;
  }
  buffer[i] = '\0';

  *len = buff_len;

  // Detect end
  if (ch == EOF && (i == 0 || ferror(fp))) {
    free(buffer);
    return NULL;
  }
  return buffer;
}

int readLines(const char* fileName, char* lines[], int max_line)
{
  FILE* file = fopen(fileName, "r");
  char* s;
  int   i = 0;
  int   n = 0;

  if (file == NULL) {
    printf("Open %s fail!\n", fileName);
    return -1;
  }

  while ((s = readLine(file, s, &n)) != NULL) {
    lines[i++] = s;
    if (i >= max_line)
      break;
  }
  fclose(file);
  return i;
}

int loadLabelName(const char* locationFilename, char* label[])
{
  printf("loadLabelName %s\n", locationFilename);
  readLines(locationFilename, label, OBJ_CLASS_NUM);
  return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
               int filterId, float threshold)
{
  for (int i = 0; i < validCount; ++i) {
    if (order[i] == -1 || classIds[i] != filterId) {
      continue;
    }
    int n = order[i];
    for (int j = i + 1; j < validCount; ++j) {
      int m = order[j];
      if (m == -1 || classIds[i] != filterId) {
        continue;
      }
      float xmin0 = outputLocations[n * 4 + 0];
      float ymin0 = outputLocations[n * 4 + 1];
      float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
      float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

      float xmin1 = outputLocations[m * 4 + 0];
      float ymin1 = outputLocations[m * 4 + 1];
      float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
      float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

      if (iou > threshold) {
        order[j] = -1;
      }
    }
  }
  return 0;
}

static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
{
  float key;
  int   key_index;
  int   low  = left;
  int   high = right;
  if (left < right) {
    key_index = indices[left];
    key       = input[left];
    while (low < high) {
      while (low < high && input[high] <= key) {
        high--;
      }
      input[low]   = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key) {
        low++;
      }
      input[high]   = input[low];
      indices[high] = indices[low];
    }
    input[low]   = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
  }
  return low;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
  float  dst_val = (f32 / scale) + zp;
  int8_t res     = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float fast_sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

double _get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

int activation_function_softmax(int8_t* src, float* dst, int length, int32_t zp, float scale)
{
    float* tmpdata = (float*)malloc(sizeof(float)*length);
    float max_value = -1000;
    for(int k = 0; k < length; k++){
      float dequt_data = deqnt_affine_to_f32(src[k], zp, scale);
      tmpdata[k] = fast_exp(dequt_data);
      if(dequt_data > max_value){
        max_value = dequt_data;
      }
      //printf("tmpdata: %d %f\n", src[k], tmpdata[k]);
    }
    //float gamma = *std::max(tmpdata, tmpdata + length);
    // int8_t alpha = *std::max(src, src + length);
    // int8_t beta = *std::min(src, src + length);
    // float f_alpha = sigmoid(deqnt_affine_to_f32(alpha, zp, scale));    
    // float f_beta = sigmoid(deqnt_affine_to_f32(beta, zp, scale));    
    // float gamma = f_alpha > f_beta ? f_alpha : f_beta;
    float denominator{ 0 };
    //printf("max_value: %f \n", max_value);

    for (int i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(deqnt_affine_to_f32(src[i], zp, scale) - max_value);
        //printf("dst[i]: %f ", dst[i]);
        denominator += dst[i];
    }
    //printf("denominator: %f \n", denominator);

    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }

    if(tmpdata)
    {
      free(tmpdata);
    }

    return 0;
}

int activation_function_softmax_finput(float* src, float* dst, int length, int32_t zp, float scale)
{
    float* tmpdata = (float*)malloc(sizeof(float)*length);
    float max_value = -10000;
    for(int k = 0; k < length; k++){
      float dequt_data = src[k];
      tmpdata[k] = fast_exp(dequt_data);
      if(dequt_data > max_value){
        max_value = dequt_data;
      }
      //printf("tmpdata: %d %f\n", src[k], tmpdata[k]);
    }
    //float gamma = *std::max(tmpdata, tmpdata + length);
    // int8_t alpha = *std::max(src, src + length);
    // int8_t beta = *std::min(src, src + length);
    // float f_alpha = sigmoid(deqnt_affine_to_f32(alpha, zp, scale));    
    // float f_beta = sigmoid(deqnt_affine_to_f32(beta, zp, scale));    
    // float gamma = f_alpha > f_beta ? f_alpha : f_beta;
    float denominator{ 0 };
    //printf("max_value: %f \n", max_value);

    for (int i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(deqnt_affine_to_f32(src[i], zp, scale) - max_value);
        //printf("dst[i]: %f ", dst[i]);
        denominator += dst[i];
    }
    //printf("denominator: %f \n", denominator);

    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }

    if(tmpdata)
    {
      free(tmpdata);
    }

    return 0;
}

static int process(int8_t* input, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
                   int32_t zp, float scale)
{
  int    Count = 0;
  int    grid_len   = grid_h * grid_w;
  // float  thres      = unsigmoid(threshold);
  // int8_t thres_i8   = qnt_f32_to_affine(thres, zp, scale);
  
  //printf("\ngrid_len h w h*w:%d %d %d %d \n", grid_len, grid_h, grid_w, grid_h*grid_w);  
  float_t unsigmoid_conf_thresh = unsigmoid(threshold);
  int8_t i8_unsigmoid_conf_thresh = qnt_f32_to_affine(unsigmoid_conf_thresh, zp, scale);
  //printf("test conf thresh:%f %f\n", threshold, sigmoid(deqnt_affine_to_f32(i8_unsigmoid_conf_thresh, zp, scale)));

  int8_t i8_conf_thresh = qnt_f32_to_affine(threshold, zp, scale);
  printf("i8tof32confthresh: %f \n", deqnt_affine_to_f32(i8_conf_thresh, zp, scale));

  for(int i = 0; i < grid_h; i++){//row
    for(int j = 0; j < grid_w; j++){//col
      //int8_t i8_max_score = -255;
      int8_t i8_max_score = -128;                                                                                                                                                                                                                                                                                  
      int cur_label = 0;
      for(int a = 0; a < OBJ_CLASS_NUM; a++){
        int32_t class_index = a*grid_len + i * grid_w + j;
        //printf("%d ", class_index);
        int8_t cls_confidence = input[class_index];         
        if(cls_confidence > i8_max_score){          
          i8_max_score = cls_confidence;          
          cur_label = a;          
        }
      }
      
      if(i8_max_score > i8_unsigmoid_conf_thresh){   
        //printf("i8_max_score: %d \n", i8_max_score);      
        float f_max_score = fast_sigmoid(deqnt_affine_to_f32(i8_max_score, zp, scale));
        //printf("f_max_score: %f \n", f_max_score);    
        float ct_x = (j+0.5) * stride;
        float ct_y = (i+0.5) * stride;
        std::vector<float> dis_pred;
        dis_pred.resize(4);  
        //printf("before for loop\n");
        for(int k = 0; k < 4; k++){    
          //printf("k:%d ", k);
          int tmp_index = 0;  
          int8_t* box_data = (int8_t*)malloc(sizeof(int8_t)*(REG_MAX + 1));
          for(int m = OBJ_CLASS_NUM + k*(REG_MAX + 1); m < (OBJ_CLASS_NUM + (k+1)*(REG_MAX + 1)); m++){
            //printf("m:%d ", m);
            int32_t box_index = m*grid_len + i*grid_w + j;            
            box_data[tmp_index] = input[box_index];
            //printf("%d %d %d %d ", tmp_index, box_index, box_data[tmp_index], input[box_index]);
            //printf("\ni j k box_index:%d %d %d %d\n", i, j, k, box_index);
            tmp_index++;            
          }
         
          float dis = 0;
          float* dis_after_sm = (float*)malloc(sizeof(float)*(REG_MAX + 1));
          activation_function_softmax(box_data, dis_after_sm, REG_MAX+1, zp, scale);
          float sum_dis = 0;
          for(int dis_index = 0; dis_index < REG_MAX + 1; dis_index++){
            dis += dis_index * dis_after_sm[dis_index];
            sum_dis += dis_after_sm[dis_index];
          }
          //printf("sum_dis: %f\n", sum_dis);
          dis *= stride;
          dis_pred[k] = dis;
          if(dis_after_sm){
            free(dis_after_sm);
          }  
          if(box_data){
            free(box_data);
          }              
        }
        //printf("ct_x_y_dispred:%f %f %f %f %f %f\n", ct_x, ct_y, dis_pred[0], dis_pred[1], dis_pred[2], dis_pred[3]);
        float xmin = (std::max)(ct_x - dis_pred[0], .0f);
        float ymin = (std::max)(ct_y - dis_pred[1], .0f);
        float xmax = (std::min)(ct_x + dis_pred[2], (float)width);
        float ymax = (std::min)(ct_y + dis_pred[3], (float)height);
        float box_width = xmax - xmin;
        float box_height = ymax - ymin;
        objProbs.push_back(f_max_score);
        classId.push_back(cur_label);        
        boxes.push_back(xmin);
        boxes.push_back(ymin);
        boxes.push_back(box_width);
        boxes.push_back(box_height);
        //printf("boxes:%f %f %f %f\n", xmin, ymin, box_width, box_height);
        Count++;
      }      
    }
  }
    
  return Count;
}

int post_process(rknn_output* outputs, int n_output, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales, detect_result_group_t* group)
{
  static int init = -1;
  if (init == -1) {
    int ret = 0;
    ret     = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0) {
      return -1;
    }
    init = 0;
  }
  memset(group, 0, sizeof(detect_result_group_t));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int>   classId;
  
  int SumCount = 0;
  struct timeval start_time, stop_time;
  for(int i = 0; i < n_output; i++)
  {
    int stride = STRIDE[i];
    int grid_h = ceil(double(model_in_h) / stride);
    int grid_w = ceil(double(model_in_w) / stride);
    int validCount = 0;
    gettimeofday(&start_time, NULL);
    validCount = process((int8_t*)outputs[i].buf, grid_h, grid_w, model_in_h, model_in_w, stride, filterBoxes, objProbs,
                         classId, conf_threshold, qnt_zps[i], qnt_scales[i]);
    gettimeofday(&stop_time, NULL);
    printf("process use %f ms\n", (_get_us(stop_time) - _get_us(start_time)) / 1000);
    SumCount += validCount;
    //printf("%d  %d  %d %d\n", stride, grid_h, grid_w, validCount);
  }
  //printf("SumCount: %d\n", SumCount);

  // // no object detect
  if (SumCount <= 0) {
    return 0;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < SumCount; ++i) {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, SumCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(SumCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  group->count   = 0;
  /* box valid detect target */
  for (int i = 0; i < SumCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];

    float x1       = filterBoxes[n * 4 + 0];
    float y1       = filterBoxes[n * 4 + 1];
    float x2       = x1 + filterBoxes[n * 4 + 2];
    float y2       = y1 + filterBoxes[n * 4 + 3];
    int   id       = classId[n];
    float obj_conf = objProbs[i];

    group->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / scale_w);
    group->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / scale_h);
    group->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / scale_w);
    group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
    group->results[last_count].prop       = obj_conf;
    char* label                           = labels[id];
    strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

    // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
    // group->results[last_count].box.top,
    //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
    last_count++;
  }
  group->count = last_count;

  return 0;
}

static int float_process(float* input, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
                   int32_t zp, float scale)
{
  int    Count = 0;
  int    grid_len   = grid_h * grid_w;
  float  thres      = unsigmoid(threshold);
  int8_t thres_i8   = qnt_f32_to_affine(thres, zp, scale);
  
  //printf("\ngrid_len h w h*w:%d %d %d %d \n", grid_len, grid_h, grid_w, grid_h*grid_w);  
  
  for(int i = 0; i < grid_h; i++){//row
    for(int j = 0; j < grid_w; j++){//col
      //int8_t i8_max_score = -255;
      float f_max_score = -10000;
      int cur_label = 0;
      for(int a = 0; a < OBJ_CLASS_NUM; a++){
        int32_t class_index = a*grid_len + i * grid_h + j;
        //printf("%d ", class_index);
        float cls_confidence = input[class_index];
        //float f_cls_conf = fast_sigmoid(cls_confidence);
        if(cls_confidence > f_max_score){
          f_max_score = cls_confidence;
          cur_label = a;          
        }
      }
      
      if(f_max_score > thres){       
        //printf("f_max_score: %f \n", f_max_score);        
        f_max_score = sigmoid(f_max_score);
        float ct_x = (j+0.5) * stride;
        float ct_y = (i+0.5) * stride;
        std::vector<float> dis_pred;
        dis_pred.resize(4);  
        //printf("before for loop\n");
        for(int k = 0; k < 4; k++){    
          //printf("k:%d ", k);
          int tmp_index = 0;  
          float* box_data = (float*)malloc(sizeof(float)*(REG_MAX + 1));
          for(int m = OBJ_CLASS_NUM + k*(REG_MAX + 1); m < (OBJ_CLASS_NUM + (k+1)*(REG_MAX + 1)); m++){
            //printf("m:%d ", m);
            int32_t box_index = m*grid_len + i*grid_h + j;            
            box_data[tmp_index] = input[box_index];
            //printf("%d %d %d %d ", tmp_index, box_index, box_data[tmp_index], input[box_index]);
            //printf("\ni j k box_index:%d %d %d %d\n", i, j, k, box_index);
            tmp_index++;            
          }
         
          float dis = 0;
          float* dis_after_sm = (float*)malloc(sizeof(float)*(REG_MAX + 1));
          activation_function_softmax_finput(box_data, dis_after_sm, REG_MAX+1, zp, scale);
          float sum_dis = 0;
          for(int dis_index = 0; dis_index < REG_MAX + 1; dis_index++){
            dis += dis_index * dis_after_sm[dis_index];
            sum_dis += dis_after_sm[dis_index];
          }
          //printf("sum_dis: %f\n", sum_dis);
          dis *= stride;
          dis_pred[k] = dis;
          if(dis_after_sm){
            free(dis_after_sm);
          }  
          if(box_data){
            free(box_data);
          }              
        }
        //printf("ct_x_y_dispred:%f %f %f %f %f %f\n", ct_x, ct_y, dis_pred[0], dis_pred[1], dis_pred[2], dis_pred[3]);
        float xmin = (std::max)(ct_x - dis_pred[0], .0f);
        float ymin = (std::max)(ct_y - dis_pred[1], .0f);
        float xmax = (std::min)(ct_x + dis_pred[2], (float)width);
        float ymax = (std::min)(ct_y + dis_pred[3], (float)height);
        float box_width = xmax - xmin;
        float box_height = ymax - ymin;
        objProbs.push_back(f_max_score);
        classId.push_back(cur_label);        
        boxes.push_back(xmin);
        boxes.push_back(ymin);
        boxes.push_back(box_width);
        boxes.push_back(box_height);
        //printf("boxes:%f %f %f %f\n", xmin, ymin, box_width, box_height);
        Count++;
      }      
    }
  }
    
  return Count;
}

int float_post_process(rknn_output* outputs, int n_output, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales, detect_result_group_t* group)
{
  static int init = -1;
  if (init == -1) {
    int ret = 0;
    ret     = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0) {
      return -1;
    }
    init = 0;
  }
  memset(group, 0, sizeof(detect_result_group_t));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int>   classId;
  
  int SumCount = 0;
  for(int i = 0; i < n_output; i++)
  {
    int stride = STRIDE[i];
    int grid_h = ceil(double(model_in_h) / stride);
    int grid_w = ceil(double(model_in_w) / stride);
    int validCount = 0;
    
    validCount = float_process((float*)outputs[i].buf, grid_h, grid_w, model_in_h, model_in_w, stride, filterBoxes, objProbs,
                         classId, conf_threshold, qnt_zps[i], qnt_scales[i]);
    SumCount += validCount;
    //printf("%d  %d  %d %d\n", stride, grid_h, grid_w, validCount);
  }
  //printf("SumCount: %d\n", SumCount);

  // // no object detect
  if (SumCount <= 0) {
    return 0;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < SumCount; ++i) {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, SumCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(SumCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  group->count   = 0;
  /* box valid detect target */
  for (int i = 0; i < SumCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];

    float x1       = filterBoxes[n * 4 + 0];
    float y1       = filterBoxes[n * 4 + 1];
    float x2       = x1 + filterBoxes[n * 4 + 2];
    float y2       = y1 + filterBoxes[n * 4 + 3];
    int   id       = classId[n];
    float obj_conf = objProbs[i];

    group->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / scale_w);
    group->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / scale_h);
    group->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / scale_w);
    group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
    group->results[last_count].prop       = obj_conf;
    char* label                           = labels[id];
    strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

    // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
    // group->results[last_count].box.top,
    //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
    last_count++;
  }
  group->count = last_count;

  return 0;
}

void deinitPostProcess()
{
  for (int i = 0; i < OBJ_CLASS_NUM; i++) {
    if (labels[i] != nullptr) {
      free(labels[i]);
      labels[i] = nullptr;
    }
  }
}
