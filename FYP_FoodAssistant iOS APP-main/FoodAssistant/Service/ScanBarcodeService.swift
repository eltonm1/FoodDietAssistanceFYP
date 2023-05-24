//
//  ScanBarcodeService.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 6/12/2022.
//

import Foundation
import AVFoundation
import UIKit
import SwiftUI
import Vision
import ImageIO

class ScanBarcodeService: NSObject, ObservableObject {
    
    /// Note:
    /// This `barcode` property can either be changed by
    /// - the functon `processClassification` whenever a new barcode has been detected by `Vision`
    /// - the textfield on the header in `CameraView`
    /// The new value will replace the old value, so the user-inputted barcode maybe replaced by newly detected
    /// barcode when `Vision` has detected a new barcode on the screen
    @Published var barcode: String = ""
    @Published var errorMessage: String?
    @Published var boundingBox: CGRect?
    @Published var normalizedBbox: CGRect?
    
    var cameraService: CameraService
    
    let detectBarcodeRequest = VNDetectBarcodesRequest()

    init(cameraService: CameraService) {
        self.cameraService = cameraService
        super.init()
        
        setUpBarcodeRequest()
    }
    
    private func setUpBarcodeOutput(
        captureSession: AVCaptureSession,
        captureOutput: AVCaptureVideoDataOutput,
        sessionQueue: DispatchQueue
    ) {
        captureOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        captureOutput.setSampleBufferDelegate(self, queue: sessionQueue)
    }
    
    private func processClassification(imageSize: CGSize) {
        guard let observations: [VNBarcodeObservation] = detectBarcodeRequest.results,
              let observation: VNBarcodeObservation = observations.first
        else {
            /// If currently there is no barcode and boundingBox stored,
            /// then there is no need to update the UI, thus return from the function
            if barcode.isEmpty || boundingBox == nil {
                return
            }
            Task {
                await MainActor.run {
                    boundingBox = nil
                    normalizedBbox = nil
                }
            }
            return
        }
            
        /// Note:
        /// Since the video preview's content mode id `.aspectFill`, so we
        /// specific the content mode to also be `.fill`, so that the coordinate of the
        /// drawn bounding box matches the container (which is the video preview)
        let convertedRect: CGRect = self.getConvertedRect(
            boundingBox: observation.boundingBox,
            inImage: imageSize,
            containedIn: UIScreen.protraitSize,
            contentMode: .fill
        )
        Task {
            await MainActor.run {
                var convertedRect = convertedRect
                var convertedNormRect = observation.boundingBox
                let orientation: UIInterfaceOrientation = UIApplication
                    .shared
                    .connectedScenes
                    .compactMap({ ($0 as? UIWindowScene)?.keyWindow })
                    .first?
                    .windowScene?
                    .interfaceOrientation ?? .portrait
                if orientation == .landscapeRight {
                    /// Rotate the rect by 90 degrees counter-clockwise
                    convertedRect = CGRect(
                        x: convertedRect.origin.y,
                        y: UIScreen.protraitSize.width - convertedRect.origin.x - convertedRect.width,
                        width: convertedRect.height,
                        height: convertedRect.width
                    )
                    convertedNormRect = CGRect(
                        x: convertedNormRect.origin.y,
                        y: 1 - convertedNormRect.origin.x - convertedNormRect.width,
                        width: convertedNormRect.height,
                        height: convertedNormRect.width
                        )
                } else if orientation == .landscapeLeft {
                    /// Rotate the rect by 90 degrees clockwise
                    convertedRect = CGRect(
                        x: UIScreen.protraitSize.height - convertedRect.origin.y - convertedRect.height,
                        y: convertedRect.origin.x,
                        width: convertedRect.height,
                        height: convertedRect.width
                    )
                    convertedNormRect = CGRect(
                        x: 1 - convertedNormRect.origin.y - convertedNormRect.height,
                        y: convertedNormRect.origin.x,
                        width: convertedNormRect.height,
                        height: convertedNormRect.width
                    )
                }
                boundingBox = convertedRect
                normalizedBbox = convertedNormRect
                
                guard let newBarcode: String = observation.payloadStringValue,
                      newBarcode != barcode
                else { return }
                barcode = newBarcode
            }
        }
    }
    
    private func setUpBarcodeRequest() {
        detectBarcodeRequest.symbologies = [.ean8, .ean13, .pdf417]
    }
    
    /// Return the appropriate image frame that best suit to be placed inside
    /// the container with specific size
    ///
    /// Suppose an image needs to be placed (either `fit` or `fill`) inside a container,
    /// but they are in different sizes. Call this function to get an image frame whose x and y
    /// well are offseted, and the returned frame size is respect to the original image aspect
    /// ratio, which can be directly used as the frame of the image placed inside the container.
    ///
    /// - Parameters:
    ///   - imageSize: image Size
    ///   - containerSize: container size in which the image to be placed on
    ///   - contentMode: the content mode that the image are placed on the container
    ///
    /// - Returns: The image rect representing the placed image's frame on the container
    func convertImage(
        ofSize imageSize: CGSize,
        to containerSize: CGSize,
        contentMode: ContentMode
    ) -> CGRect {
        let rectOfImage: CGRect
        let imageAspect = imageSize.width / imageSize.height // 1.78
        let containerAspect = containerSize.width / containerSize.height // 0.46
        
        if contentMode == .fill {
            if imageAspect > containerAspect { /// image extends left and right
                let newImageWidth = containerSize.height * imageAspect /// the width of the overflowing image
                let newX = -(newImageWidth - containerSize.width) / 2
                rectOfImage = CGRect(x: newX, y: 0, width: newImageWidth, height: containerSize.height)
                
            } else { /// image extends top and bottom
                let newImageHeight = containerSize.width * (1 / imageAspect) /// the width of the overflowing image
                let newY = -(newImageHeight - containerSize.height) / 2
                rectOfImage = CGRect(x: 0, y: newY, width: containerSize.width, height: newImageHeight)
            }
            
            return rectOfImage
        } else {
            if imageAspect > containerAspect { /// container extends top and bottom
                let newImageHeight = containerSize.width * (1 / imageAspect) /// the width of the overflowing image
                let newY = (containerSize.height - newImageHeight) / 2
                rectOfImage = CGRect(x: 0, y: newY, width: containerSize.width, height: newImageHeight)
            } else { /// container extends left and right
                let newImageWidth = containerSize.height * imageAspect /// the width of the overflowing image
                let newX = (containerSize.width - newImageWidth) / 2
                rectOfImage = CGRect(x: newX, y: 0, width: newImageWidth, height: containerSize.height)
            }

            return rectOfImage
        }
    }
    
    /// Convert the normalized bounding box to bounding box whose coordinate system
    /// is the same as the the container
    ///
    /// The `boundingBox` returned from `Vision` is normalized w.r.t to image size, and its
    /// origin is at the top-left corner of the image, while the normal UI coordinate system 
    /// is at the bottom-left corner of the screen. In order to draw the bounding box on the 
    /// screen, one needs to transform the normalized `boundingBox` to back to the normal.
    ///
    /// - Parameters:
    ///    - boundingBox: a `CGRect` returned from `Vision` framework
    ///    - inImage: a `CGSize` of the image
    ///    - containedIn: a `CGSize` of the container which the image is displayed on. In our project case,
    ///    This is the size of the full screen, because the image is displayed on `CameraView`, and `CameraView`
    ///    is taking the full screen size.
    ///    - contentMode: specify the content mode of the image to be displayed (either `.fill` or `.fit`)
    /// 
    /// - Returns: a `CGRect` whose coordinate system is the same as the container
    func getConvertedRect(
        boundingBox: CGRect,
        /// 1920x1080 when photo is captured by camera (other size in otherwise,
        /// e.g. Photo library, which can have custom size)
        inImage imageSize: CGSize,
        containedIn containerSize: CGSize, // 428x926 or 926x428,
        contentMode: ContentMode = .fill
    ) -> CGRect {
        let rectOfImage: CGRect = convertImage(ofSize: imageSize, to: containerSize, contentMode: contentMode)
        return transform(normalizedBBox: boundingBox, to: rectOfImage)
    }
    
    /// Convert the normalized bounding box to bounding box whose coordinate system
    /// is the same as the image rect provided
    ///
    /// The `boundingBox` returned from `Vision` is normalized w.r.t to image size, and its
    /// origin is at the top-left corner of the image, while the normal UI coordinate system
    /// is at the bottom-left corner of the screen. In order to draw the bounding box on the
    /// screen, one needs to transform the normalized `boundingBox` to back to the normal.
    ///
    /// - Parameters:
    ///    - normalizedBBox: a `CGRect` returned from `Vision` framework, in which bounding box
    ///    is normalized w.r.t. image size
    ///    - rectOfImage: a appropriate image frame that represent the image frame when placed inside
    ///    the container
    ///
    /// - Returns: a `CGRect` whose coordinate system is the same as the container
    func transform(normalizedBBox boundingBox: CGRect, to rectOfImage: CGRect) -> CGRect {
        let newOriginBoundingBox = CGRect(
            x: boundingBox.origin.x,
            y: 1 - boundingBox.origin.y - boundingBox.height,
            width: boundingBox.width,
            height: boundingBox.height
        )
        
        var convertedRect = VNImageRectForNormalizedRect(newOriginBoundingBox, Int(rectOfImage.width), Int(rectOfImage.height))
        
        /// add the margins
        convertedRect.origin.x += rectOfImage.origin.x
        convertedRect.origin.y += rectOfImage.origin.y
        
        return convertedRect
    }
    
    func onAppear() {
        cameraService.configure(additionalOutput: setUpBarcodeOutput)
    }
    
    func onDisappear() {
        cameraService.stop()
    }
    
    /// Detect barcode from an image
    ///
    /// This function is called when user capture the photo (either from camera or photo library via image picker)
    /// 
    /// - Parameters:
    ///   - image: An UIImage object to be detected from
    ///   - source: The source where the image is captured from.
    ///
    func detectBarcode(from image: UIImage, on source: CaptureSource) {
        guard let cgImage: CGImage = image.cgImage else { return }
        let imageOrientation: UIImage.Orientation = image.imageOrientation
        let cgOrientation: CGImagePropertyOrientation
        if source == .byCamera {
            switch imageOrientation {
            case .right: // Portrait mode
                cgOrientation = .right
                print("Camera Portrait mode")
            case .down: // Landscape Right
                cgOrientation = .left
                print("Camera Landscape Right")
            case .up: // Landscape Left
                cgOrientation = .left
                print("Camera Landscape Left")
            default:
                cgOrientation = .right
                print("Camera unknown")
            }
        } else {
            switch imageOrientation {
            case .right: // Portrait mode
                cgOrientation = .right
                print("ImagePicker Portrait mode")
            case .down: // Landscape Right
                cgOrientation = .down
                print("ImagePicker Landscape Right")
            case .up: // Landscape Left
                cgOrientation = .up //.right //.left
                print("ImagePicker Landscape Left")
            case .left:
                cgOrientation = .left
            default:
                cgOrientation = .right
                print("ImagePicker unknown")
            }
        }

        let requestHandler = VNImageRequestHandler(
            cgImage: cgImage,
            orientation: cgOrientation, // CGImagePropertyOrientation(imageOrientation),
            options: [:]
        )
        
        do {
            try requestHandler.perform([detectBarcodeRequest])
        } catch {
            print(error)
            errorMessage = error.localizedDescription
        }
        
        processClassification(
            imageSize: image.size
        )
    }
    
    deinit {
        print("deinited ScanBarcodeService")
    }
    
}

extension ScanBarcodeService: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        /// `CVPixelBuffer` stands for `Core Video Pixel Buffer`
        guard let pixelBuffer: CVImageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
//        let uiImage = UIImage(ciImage: ciImage, scale: 1.0, orientation: .right)
//        let imageRequestHandler = VNImageRequestHandler(
//          cvPixelBuffer: pixelBuffer,
//          orientation: .right
//        )
        
        let uiImage = UIImage(ciImage: ciImage, scale: 1.0, orientation: .right)
        let imageRequestHandler = VNImageRequestHandler(
          cvPixelBuffer: pixelBuffer,
          orientation: .right
        )
        
        do {
          try imageRequestHandler.perform([detectBarcodeRequest])
        } catch {
            print(error.localizedDescription)
            self.errorMessage = error.localizedDescription
        }
        
        processClassification(imageSize: uiImage.size)
    }
}
