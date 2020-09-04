#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#include <TesseractOCR/TesseractOCR.h>


NS_ASSUME_NONNULL_BEGIN

@interface OpenCVWrapper : NSObject <G8TesseractDelegate>

+(UIImage*)readMRZ:(UIImage*)image;
+(CGRect)getMRZRect;
+(NSString*)getMRZText;

@end

NS_ASSUME_NONNULL_END
