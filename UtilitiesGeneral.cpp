// General routines for reading, writing
// and displaying images.

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>
#include "tinyfiledialogs.h"
#include <stdio.h>
#include <time.h>
#include <iostream>
#include "windows.h"

int DisplayImage(cv::Mat image,const std::string &windowname );
std::string toUpper(const std::string& s);
std::string toUpperThenLower(const std::string& s);

// Michele: Boring utilities.
// For a Photoshop plugin the main image would be the one on display
// the palette image would already have been read in and would
// need to be selected from a list of already loaded images
// As in the Photoshop function Match Colour at present.


cv::Mat GetImage(const std::string &filetype)
{
// Select image file for input, read it in, then display it.
    cv::Mat image;
    std::string winheader;
    char const * SelectedFile;
    int imagegood=-1;

    winheader = "SELECT FILE FOR INPUT AS "+toUpper(filetype)+" IMAGE";
    const char * windowheader = winheader.c_str();

    while(imagegood<0)
    {
        // Select file for input. Uses "tiny_file_dialogs" to achieve this.
        SelectedFile = tinyfd_openFileDialog(windowheader
                  ,"./*.jpg;*.bmp;*.tif;*.tif;*.png", 0, NULL , NULL , 0);

        if (SelectedFile==NULL)
        {
           SelectedFile="C:/NonExistentFile";
        }

        //Read in the image.
        image = cv::imread(SelectedFile, 1);

        //Set up window header and then display image.
        winheader = toUpperThenLower(filetype) + " Image";
        windowheader = winheader.c_str();
        imagegood=DisplayImage(image, windowheader);
        if(imagegood<0)
        {
            tinyfd_messageBox("Image Not Captured",
            "Cannot open selected file in reading mode. Try again",
            "ok", "error", 1 );
        }
    }
    return image;
}


void SaveImage(cv::Mat image, const std::string &filetype, bool fordisplay, bool forsave, int stage)
{
// Display image if required and then output image to selected file.
    std::string winheader;
    char stagelookup[3]={char(0),'1','2'};
    char const * (SelectedFile);
    std::string filename=filetype+stagelookup[stage];
    if (fordisplay)
    {
        DisplayImage(image, filename);
    }
    if(forsave)
    {
        winheader = "SELECT OUTPUT FILE FOR "+toUpper(filetype)+" IMAGE";
        const char * windowheader = winheader.c_str();
        winheader = "./"+toUpperThenLower(filetype)+".jpg";
        const char * fileselect = winheader.c_str();

        SelectedFile = tinyfd_saveFileDialog(windowheader ,
                                    fileselect ,0 , NULL , NULL);
       

        // Write the image to file
        if(SelectedFile!=NULL)
        {
            cv::imwrite(SelectedFile, image);
        }
    }
    return;
}


int DisplayImage(cv::Mat image,const std::string &windowname )
{
// Display the supplied image using a designated window title.

    // Create a window for display of dimensions 640x480 and display the image.
    cv::namedWindow(windowname,cv::WINDOW_AUTOSIZE);
    cv::Mat img;
    if(image.cols>image.rows)
    {
        cv::resize(image,img,cv::Size(640,480),0,0,cv::INTER_AREA);
    }
    else
    {
        cv::resize(image,img,cv::Size(480,640),0,0,cv::INTER_AREA);
    }
    cv::imshow(windowname, img);
    return 0;
}

/*
cv::Mat AddWatermark(cv::Mat image,const std::string &watermarktext)
{
//If watermarktext is not null, then add the text as a watermark to
//the input image.

    cv::Mat wimage=image.clone();

    if(watermarktext==""){}
    else
    {
        float scale=image.cols/512.0;
        cv::putText(wimage,watermarktext,cv::Point(image.cols*0.1,image.rows*0.9),
                    cv::FONT_HERSHEY_COMPLEX_SMALL,1*scale,cv::Scalar(0,255,255),1*scale);
        image=0.90*image+0.1*wimage;
    }
    return image;
}

std::string OpeningScreen()
{
 // Displays opening screen and implements date limit if any.

    int year=2022;
    int month=12;
    int day=31;

    time_t timenow;
    struct tm settime;
    double seconds;
    char str[100];
    cv::Mat image;
    std::string strout="";

    time(&timenow);
    settime = *localtime(&timenow);
    settime.tm_year=year-1900;
    settime.tm_mon=month-1;
    settime.tm_mday=day;

    seconds=difftime(mktime(&settime),timenow);


    if (year==0)
    {
        sprintf(str,"NOT FOR RESALE.  Copyright Terry Johnson December 2015.");
        image=cv::imread("Title2.jpg");
        // Display title page and go
        DisplayImage(image,str);
        cv::waitKey(700);
    }
    else if(seconds>=0.0)
    {
        sprintf(str,"NOT FOR RESALE.  EXPIRES ON %d - %d - %d   Copyright Terry Johnson December 2015.",
                    day,month,year);
        image=cv::imread("Title1.jpg");
        // Display expiry date and wait for key input.
         DisplayImage(image,str);
         cv::waitKey(10000);
    }
    else
    {
            image=cv::imread("Expired.jpg");
            strout="licence expired wait for 5 minutes";
            // Out of date add a substantial wait
            // before proceeding.
            DisplayImage(image,str);
            cv::waitKey(1);
            Sleep(180000);
    }
return strout;
}*/


std::string toUpper(const std::string& s)
{
// Convert string to upper case.
    std::string result;

    for (int i = 0; i < s.length(); i++)
    {
        result += ::toupper(s.at(i));
    }
    return result;
}


std::string toUpperThenLower(const std::string& s)
{
// Convert first character of string to upper case
// and the rest to lower case.
    std::string result;

    result= ::toupper(s.at(0));

    for (int i = 1; i < s.length(); i++)
    {
        result += ::tolower(s.at(i));
    }
    return result;
}



