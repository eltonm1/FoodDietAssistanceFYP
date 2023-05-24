//
//  NutritionTableDetectionService.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 4/12/2022.
//

import Foundation
import Vision
import UIKit

class NutritionTableDetectionService: ObservableObject {
    
    @Published var boundingBox: BoundingBox? = nil
    @Published var model: VNCoreMLModel?
    
    private var imageToBeDetected: UIImage?
    
    init() {
        Task(priority: .userInitiated) { [weak self] in
            print("Initializing YOLOv4")
            let model: VNCoreMLModel? = try? VNCoreMLModel(for: YOLOv4(configuration: .init()).model)
            await MainActor.run { [weak self] in
                self?.model = model
                print("Finished initializing YOLOv4")
            }
        }
    }
    
    func detectNuritionTable(image: UIImage) {
        guard let model else { return }
        let request = VNCoreMLRequest(model: model, completionHandler: visionRequestDidComplete)
        request.imageCropAndScaleOption = .scaleFill
        Task {
            var orientation: CGImagePropertyOrientation = .up
            if image.imageOrientation == .right {
                orientation = .right
            } else if image.imageOrientation == .left {
                orientation = .left
            }
            let handler = VNImageRequestHandler(cgImage: image.cgImage!, orientation: orientation)//options: [:]
            do {
                try handler.perform([request])
                imageToBeDetected = image
            } catch {
                print(error)
            }
        }
    }
    
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNCoreMLFeatureValueObservation],
              let bbox = observations.first?.featureValue.multiArrayValue else { return }
        let bboxShapedArray = MLShapedArray<Float>(bbox)
        let predictions: [BoundingBox] = bboxShapedArray
            .filter { (output: MLShapedArraySlice<Float>) in
                output.scalars[4] > 0 // 0.5
            }
            .map {
                (output: MLShapedArraySlice<Float>) -> BoundingBox in
                let x = output.scalars
                return BoundingBox(
                    classIndex: Int(x[5]),
                    score: x[4],
                    rect: CGRect(
                        x: Int(x[0]),
                        y: Int(x[1]),
                        width: Int(x[2]),
                        height: Int(x[3])
                    )
                )
            }
        
        let nms = NMS(bboxes: predictions, iouThreshold: 0.5, maxBoxes: 100)
        let predictionsOut: BoundingBox? = nms().map { predictions[$0] }.first
        Task { [weak self] in
            await MainActor.run { [weak self] in
                self?.boundingBox = predictionsOut
            }
        }
    }
    
    /// Crop the nutrition table from the 416x416 resized image
    ///
    /// - Returns: The cropped UIImage to be sent to the backend server for 2nd model processing
    func cropTable() -> UIImage? {
        guard let bbox: BoundingBox = boundingBox,
              let image: UIImage = imageToBeDetected
        else { return nil }
        let scale: CGFloat = image.scale
        let xOffset: CGFloat = bbox.rect.width/2
        let yOffset: CGFloat = bbox.rect.height/2
        /// - Note:
        /// Because the format of bbox is [midx, midy, width, height],
        /// so we are going to subtract the [midx, midy] with its [width/2, height/2]
        /// so that it becomes [xmin, ymin, width, height]
        let scaledBBoxRect: CGRect = .init(
            x: (bbox.rect.origin.x - xOffset) * scale,
            y: (bbox.rect.origin.y - yOffset) * scale,
            width: bbox.rect.width * scale,
            height: bbox.rect.height * scale
        )
        guard let imageRef: CGImage = image.cgImage?.cropping(to: scaledBBoxRect) else { return nil }
        return UIImage(cgImage: imageRef, scale: image.scale, orientation: image.imageOrientation)
    }
}
