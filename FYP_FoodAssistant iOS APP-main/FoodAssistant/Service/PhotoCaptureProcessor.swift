//
//  PhotoCaptureProcessor.swift
//  abseil
//
//  Created by Rolando Rodriguez on 1/11/20.
//

import Foundation
import Photos

class PhotoCaptureProcessor: NSObject {
    private(set) var requestedPhotoSettings: AVCapturePhotoSettings
    
    private let willCapturePhotoAnimation: () -> Void
    
    lazy var context = CIContext()
    
    private let completionHandler: (PhotoCaptureProcessor) -> Void
    
    private let photoProcessingHandler: (Bool) -> Void
    
    var photoData: Data?
    
    private var maxPhotoProcessingTime: CMTime?
    
    init(
        with requestedPhotoSettings: AVCapturePhotoSettings,
        willCapturePhotoAnimation: @escaping () -> Void,
        completionHandler: @escaping (PhotoCaptureProcessor) -> Void,
        photoProcessingHandler: @escaping (Bool) -> Void
    ) {
        self.requestedPhotoSettings = requestedPhotoSettings
        self.willCapturePhotoAnimation = willCapturePhotoAnimation
        self.completionHandler = completionHandler
        self.photoProcessingHandler = photoProcessingHandler
    }
}

extension PhotoCaptureProcessor: AVCapturePhotoCaptureDelegate {
    /*
     This extension adopts all of the AVCapturePhotoCaptureDelegate protocol methods.
     */
    
    /// - Tag: WillBeginCapture
    func photoOutput(_ output: AVCapturePhotoOutput, willBeginCaptureFor resolvedSettings: AVCaptureResolvedPhotoSettings) {
        maxPhotoProcessingTime = resolvedSettings.photoProcessingTimeRange.start + resolvedSettings.photoProcessingTimeRange.duration
    }
    
    /// - Tag: WillCapturePhoto
    func photoOutput(_ output: AVCapturePhotoOutput, willCapturePhotoFor resolvedSettings: AVCaptureResolvedPhotoSettings) {
        DispatchQueue.main.async { [weak self] in
            self?.willCapturePhotoAnimation()
        }
        
        guard let maxPhotoProcessingTime else {
            return
        }
        
        // Show a spinner if processing time exceeds one second.
        let oneSecond = CMTime(seconds: 2, preferredTimescale: 1)
        if maxPhotoProcessingTime > oneSecond {
            DispatchQueue.main.async { [weak self] in
                self?.photoProcessingHandler(true)
            }
        }
    }
    
    /// - Tag: DidFinishProcessingPhoto
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        
        DispatchQueue.main.async { [weak self] in
            self?.photoProcessingHandler(false)
        }
        
        if let error {
            print("Error capturing photo: \(error)")
        } else {
            photoData = photo.fileDataRepresentation()
            
        }
    }
    
    fileprivate func saveToPhotoLibrary(_ photoData: Data) {
        //        MARK: Saves capture to photo library
        PHPhotoLibrary.requestAuthorization { [weak self] (status: PHAuthorizationStatus) in
            guard let self = self else { return }
            if status == .authorized {
                PHPhotoLibrary.shared().performChanges {
                    let options = PHAssetResourceCreationOptions()
                    let creationRequest = PHAssetCreationRequest.forAsset()
                    options.uniformTypeIdentifier = self.requestedPhotoSettings.processedFileType.map { $0.rawValue }
                    creationRequest.addResource(with: .photo, data: photoData, options: options)
                    
                    
                } completionHandler: { [weak self] (_, error: Error?) in
                    if let error {
                        print("Error occurred while saving photo to photo library: \(error)")
                    }
                    
                    DispatchQueue.main.async { [weak self] in
                        guard let self = self else { return }
                        self.completionHandler(self)
                    }
                }
            } else {
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.completionHandler(self)
                }
            }
        }
    }
    
    /// - Tag: DidFinishCapture
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishCaptureFor resolvedSettings: AVCaptureResolvedPhotoSettings, error: Error?) {
        if let error {
            print("Error capturing photo: \(error)")
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.completionHandler(self)
            }
            return
        }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.completionHandler(self)
        }
        
        //        self.saveToPhotoLibrary(photoData)
    }
}
