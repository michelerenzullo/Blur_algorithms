#ifndef RECOLOUR_H_INCLUDED
#define RECOLOUR_H_INCLUDED
// ColourgramProcessing.cpp
cv::Mat ComputeColourgram(cv::Mat& image, int stage,
    bool colourgramdisplay, bool colourgramsave);
void Reassert(cv::Mat image, cv::Mat originalmonoimage,
    cv::Mat image2, int stage);


// UtilitiesRecolour.cpp
cv::Mat IndexFill(cv::Mat Array);
void ShuffleImage(cv::Mat& indexarray, cv::Mat& image, double factor);
void DetermineScaling(cv::Mat image, int& rows, int& cols);
void MinMaxImageVals(cv::Mat imagein, float& minout, float& maxout);
cv::Mat SwapChannels(cv::Mat image, int one, int two, int three);
void SetUpProcessingParameters();

// TwoDProcessing.cpp
void ComputeColourgram2D(cv::Mat& image, cv::Mat& indexarray, int stage);

// SortingRoutines.cpp
cv::Mat InverseSorting(cv::Mat indexarray, cv::Mat image);
void quickSortI(uint8_t* weights, signed long* indices,
    uint8_t* B, uint8_t* G, uint8_t* R,
    int low, int high);

// ColourConditioning.cpp
cv::Mat ShadingAdjustment(cv::Mat monoimage, cv::Mat refimage);
cv::Mat ColourAdjustment(cv::Mat image1, cv::Mat image2);
void ReassertOnlyProcessing(int reassertonly);
cv::Mat LMK(cv::Mat imgt, cv::Mat imgs, int mode);

// UtilitiesGeneral.cpp
cv::Mat GetImage(const std::string& filetype);
void SaveImage(cv::Mat image, const std::string& filetype, bool fordisplay, bool forsave, int stage);
//cv::Mat AddWatermark(cv::Mat image,const std::string &watermarktext);
//std::string OpeningScreen();

// DECLARE EXTERNAL VARIABLES.
// Not generally considered good practice but here
// it is cleaner to declare the input processing
// parameters as global variables rather than pass
// them from one routine to another.
extern int ProcessingMode, AdjustColour,
PreMatch;
extern bool  OptimisedForSpeed;
extern float DeSpk, PercentTint, PercentShading;

#endif // RECOLOUR_H_INCLUDED
