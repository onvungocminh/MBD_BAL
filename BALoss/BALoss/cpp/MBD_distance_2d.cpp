#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include "util.h"
#include "MBD_distance_2d.h"
#include <queue>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h> 
#include <limits.h>
using namespace std;



void MBD_cut(const unsigned char * img, const unsigned char * seeds, const unsigned char * destination, unsigned char * distance, 
                              int height, int width)
{
    // Inside
    int * state = new int[height * width];
    unsigned char * min_image = new unsigned char[height * width];
    unsigned char * max_image = new unsigned char[height * width];
    unsigned char * distance_in = new unsigned char[height * width]; 
    unsigned char * distance_out = new unsigned char[height * width]; 
    unsigned char * segment = new unsigned char[height * width]; 
    unsigned char * dejavu = new unsigned char[height * width];
    unsigned char * post_segment = new unsigned char[height * width];

    vector<queue<Point2D> > Q(256);
    queue<Point2D> Q_segment;

    // point state: 0--acceptd, 1--temporary, 2--far away
    // get initial accepted set and far away set

    unsigned char init_dis;
    int init_state;

    for(int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            Point2D p;
            p.h = h;
            p.w = w;
            unsigned char seed_type = get_pixel<unsigned char>(seeds, height, width, h, w);
            unsigned char img_value = get_pixel<unsigned char>(img, height, width, h, w);

            if(seed_type > 100){
                init_dis = 0;
                init_state = 1;
                Q[init_dis].push(p);
                Q_segment.push(p);
                set_pixel<unsigned char>(post_segment, height, width, h, w, 255);
                set_pixel<unsigned char>(distance, height, width, h, w, 0);
                set_pixel<unsigned char>(dejavu, height, width, h, w, 1);
                set_pixel<unsigned char>(distance_in, height, width, h, w, init_dis);
                set_pixel<int>(state, height, width, h, w, init_state);
                set_pixel<unsigned char>(min_image, height, width, h, w, img_value);
                set_pixel<unsigned char>(max_image, height, width, h, w, img_value);                    
            }
            else{
                set_pixel<unsigned char>(dejavu, height, width, h, w, 0);
                set_pixel<unsigned char>(distance, height, width, h, w, 0);
                set_pixel<unsigned char>(post_segment, height, width, h, w, 0);
                init_dis = 255;
                init_state = 0;
                set_pixel<unsigned char>(distance_in, height, width, h, w, init_dis);
                set_pixel<int>(state, height, width, h, w, init_state);  
                set_pixel<unsigned char>(min_image, height, width, h, w, img_value);
                set_pixel<unsigned char>(max_image, height, width, h, w, img_value);                                        
            }
        }
    }


    int dh[4] = { 1 ,-1 , 0, 0};
    int dw[4] = { 0 , 0 , 1,-1};

    // Proceed the propagation from the marker to all pixels in the image
    for (int lvl = 0; lvl < 256; lvl++)
    {

        while (!Q[lvl].empty())
        {
            Point2D p = Q[lvl].front();
            Q[lvl].pop();

            int state_value = get_pixel<int>(state, height, width, p.h, p.w);
            if (state_value == 2)
                continue;

            set_pixel<int>(state, height, width, p.h, p.w, 2);


            for (int n1 = 0 ; n1 < 4 ; n1++)
            {
                int tmp_h  = p.h + dh[n1];
                int tmp_w  = p.w + dw[n1];

                if (tmp_h >= 0 and tmp_h < height and tmp_w >= 0 and tmp_w < width)
                {
                    Point2D r;
                    r.h = tmp_h;
                    r.w = tmp_w;

                    unsigned char temp_r = get_pixel<unsigned char>(distance_in, height, width,  r.h, r.w);
                    unsigned char temp_p = get_pixel<unsigned char>(distance_in, height, width,  p.h, p.w);

                    state_value = get_pixel<int>(state, height, width, r.h, r.w);

                    if (state_value == 1 && temp_r> temp_p)
                    {
                        unsigned char min_image_value = get_pixel<unsigned char>(min_image, height, width, p.h, p.w);
                        unsigned char max_image_value = get_pixel<unsigned char>(max_image, height, width, p.h, p.w);

                        set_pixel<unsigned char>(min_image, height, width, r.h, r.w, min_image_value);
                        set_pixel<unsigned char>(max_image, height, width, r.h, r.w, max_image_value);

                        unsigned char image_value = get_pixel<unsigned char>(img, height, width, r.h, r.w);

                        if (image_value < min_image_value)
                            set_pixel<unsigned char>(min_image, height, width, r.h, r.w, image_value);
                        if (image_value > max_image_value)
                            set_pixel<unsigned char>(max_image, height, width, r.h, r.w, image_value);

                        temp_r = get_pixel<unsigned char>(distance_in, height, width, r.h, r.w);

                        min_image_value = get_pixel<unsigned char>(min_image, height, width, r.h, r.w);
                        max_image_value = get_pixel<unsigned char>(max_image, height, width, r.h, r.w);

                        unsigned char temp_dis = max_image_value - min_image_value;

                        if (temp_r > temp_dis)
                        {
                            set_pixel<unsigned char>(distance_in, height, width, r.h, r.w, temp_dis);
                            Q[temp_dis].push(r);
                        }
                    }

                    else if (state_value == 0)
                    {
                        unsigned char min_image_value = get_pixel<unsigned char>(min_image, height, width, p.h, p.w);
                        unsigned char max_image_value = get_pixel<unsigned char>(max_image, height, width, p.h, p.w);
                        set_pixel<unsigned char>(min_image, height, width, r.h, r.w, min_image_value);
                        set_pixel<unsigned char>(max_image, height, width, r.h, r.w, max_image_value);

                        unsigned char image_value = get_pixel<unsigned char>(img, height, width, r.h, r.w);

                        if (image_value < min_image_value)
                            set_pixel<unsigned char>(min_image, height, width, r.h, r.w, image_value);
                        if (image_value > max_image_value)
                            set_pixel<unsigned char>(max_image, height, width, r.h, r.w, image_value);

                        min_image_value = get_pixel<unsigned char>(min_image, height, width, r.h, r.w);
                        max_image_value = get_pixel<unsigned char>(max_image, height, width, r.h, r.w);

                        unsigned char temp_dis = max_image_value - min_image_value;

                        set_pixel<unsigned char>(distance_in, height, width, r.h, r.w, temp_dis);                   
                        Q[temp_dis].push(r);
                        set_pixel<int>(state, height, width, r.h, r.w, 1);
                    }
                    else
                        continue;

                }
            } 
        }
    }


    // Outside

    vector<queue<Point2D> > Q1(256);

    // point state: 0--acceptd, 1--temporary, 2--far away
    // get initial accepted set and far away set

    // unsigned char init_dis;
    // int init_state;

    for(int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            Point2D p;
            p.h = h;
            p.w = w;
            unsigned char seed_type = get_pixel<unsigned char>(destination, height, width, h, w);
            unsigned char img_value = get_pixel<unsigned char>(img, height, width, h, w);

            if(seed_type > 100){
                init_dis = 0;
                init_state = 1;
                Q1[init_dis].push(p);
                set_pixel<unsigned char>(distance_out, height, width, h, w, init_dis);
                set_pixel<int>(state, height, width, h, w, init_state);
                set_pixel<unsigned char>(min_image, height, width, h, w, img_value);
                set_pixel<unsigned char>(max_image, height, width, h, w, img_value);                    
            }
            else{
                init_dis = 255;
                init_state = 0;
                set_pixel<unsigned char>(distance_out, height, width, h, w, init_dis);
                set_pixel<int>(state, height, width, h, w, init_state);  
                set_pixel<unsigned char>(min_image, height, width, h, w, img_value);
                set_pixel<unsigned char>(max_image, height, width, h, w, img_value);                                        
            }
        }
    }




    // Proceed the propagation from the marker to all pixels in the image
    for (int lvl = 0; lvl < 256; lvl++)
    {

        while (!Q1[lvl].empty())
        {
            Point2D p = Q1[lvl].front();
            Q1[lvl].pop();

            int state_value = get_pixel<int>(state, height, width, p.h, p.w);
            if (state_value == 2)
                continue;

            set_pixel<int>(state, height, width, p.h, p.w, 2);


            for (int n1 = 0 ; n1 < 4 ; n1++)
            {
                int tmp_h  = p.h + dh[n1];
                int tmp_w  = p.w + dw[n1];

                if (tmp_h >= 0 and tmp_h < height and tmp_w >= 0 and tmp_w < width)
                {
                    Point2D r;
                    r.h = tmp_h;
                    r.w = tmp_w;

                    unsigned char temp_r = get_pixel<unsigned char>(distance_out, height, width,  r.h, r.w);
                    unsigned char temp_p = get_pixel<unsigned char>(distance_out, height, width,  p.h, p.w);

                    state_value = get_pixel<int>(state, height, width, r.h, r.w);

                    if (state_value == 1 && temp_r> temp_p)
                    {
                        unsigned char min_image_value = get_pixel<unsigned char>(min_image, height, width, p.h, p.w);
                        unsigned char max_image_value = get_pixel<unsigned char>(max_image, height, width, p.h, p.w);

                        set_pixel<unsigned char>(min_image, height, width, r.h, r.w, min_image_value);
                        set_pixel<unsigned char>(max_image, height, width, r.h, r.w, max_image_value);

                        unsigned char image_value = get_pixel<unsigned char>(img, height, width, r.h, r.w);

                        if (image_value < min_image_value)
                            set_pixel<unsigned char>(min_image, height, width, r.h, r.w, image_value);
                        if (image_value > max_image_value)
                            set_pixel<unsigned char>(max_image, height, width, r.h, r.w, image_value);

                        temp_r = get_pixel<unsigned char>(distance_out, height, width, r.h, r.w);

                        min_image_value = get_pixel<unsigned char>(min_image, height, width, r.h, r.w);
                        max_image_value = get_pixel<unsigned char>(max_image, height, width, r.h, r.w);

                        unsigned char temp_dis = max_image_value - min_image_value;

                        if (temp_r > temp_dis)
                        {
                            set_pixel<unsigned char>(distance_out, height, width, r.h, r.w, temp_dis);
                            Q1[temp_dis].push(r);
                        }
                    }

                    else if (state_value == 0)
                    {
                        unsigned char min_image_value = get_pixel<unsigned char>(min_image, height, width, p.h, p.w);
                        unsigned char max_image_value = get_pixel<unsigned char>(max_image, height, width, p.h, p.w);
                        set_pixel<unsigned char>(min_image, height, width, r.h, r.w, min_image_value);
                        set_pixel<unsigned char>(max_image, height, width, r.h, r.w, max_image_value);

                        unsigned char image_value = get_pixel<unsigned char>(img, height, width, r.h, r.w);

                        if (image_value < min_image_value)
                            set_pixel<unsigned char>(min_image, height, width, r.h, r.w, image_value);
                        if (image_value > max_image_value)
                            set_pixel<unsigned char>(max_image, height, width, r.h, r.w, image_value);

                        min_image_value = get_pixel<unsigned char>(min_image, height, width, r.h, r.w);
                        max_image_value = get_pixel<unsigned char>(max_image, height, width, r.h, r.w);

                        unsigned char temp_dis = max_image_value - min_image_value;

                        set_pixel<unsigned char>(distance_out, height, width, r.h, r.w, temp_dis);                   
                        Q1[temp_dis].push(r);
                        set_pixel<int>(state, height, width, r.h, r.w, 1);
                    }
                    else
                        continue;

                }
            } 
        }
    }


    // segment region  by  comparing the  two distance maps

    for(int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            unsigned char distance1 = get_pixel<unsigned char>(distance_in, height, width, h, w);
            unsigned char distance2 = get_pixel<unsigned char>(distance_out, height, width, h, w);
            if (distance1 < distance2 )
                set_pixel<unsigned char>(segment, height, width, h, w, 255);
            else
                set_pixel<unsigned char>(segment, height, width, h, w, 0);
        }
    } 

    // Post processing to select only one connected component from the seed point of th foreground

    while (!Q_segment.empty())
    {
        Point2D p = Q_segment.front();
        Q_segment.pop();
        set_pixel<unsigned char>(post_segment, height, width, p.h, p.w, 255);

        for (int n1 = 0 ; n1 < 4 ; n1++)
        {
            int tmp_h  = p.h + dh[n1];
            int tmp_w  = p.w + dw[n1];

            if (tmp_h >= 0 and tmp_h < height and tmp_w >= 0 and tmp_w < width)
            {
                Point2D r;
                r.h = tmp_h;
                r.w = tmp_w;
                unsigned char temp_r = get_pixel<unsigned char>(segment, height, width,  r.h, r.w);
                unsigned char temp_djv = get_pixel<unsigned char>(dejavu, height, width,  r.h, r.w);
                if (temp_djv == 0 and temp_r == 255)
                {
                    set_pixel<unsigned char>(dejavu, height, width, r.h, r.w, 1);
                    Q_segment.push(r);
                }
            }
        }
    }

    for(int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            Point2D p;
            p.h = h;
            p.w = w;
            for (int n1 = 0 ; n1 < 4 ; n1++)
            {
                int tmp_h  = p.h + dh[n1];
                int tmp_w  = p.w + dw[n1];

                if (tmp_h >= 0 and tmp_h < height and tmp_w >= 0 and tmp_w < width)
                {
                    unsigned char temp_p = get_pixel<unsigned char>(post_segment, height, width,  p.h, p.w);
                    unsigned char temp_r = get_pixel<unsigned char>(post_segment, height, width,  tmp_h, tmp_w);
                    if (temp_p == 0 and temp_r == 255)
                        set_pixel<unsigned char>(distance, height, width, p.h, p.w, 255);
                }
            }
        }
    }


    delete state;
    delete min_image;
    delete max_image;
    delete distance_in;
    delete distance_out;
    delete segment;
    delete dejavu;
    delete post_segment;

}

