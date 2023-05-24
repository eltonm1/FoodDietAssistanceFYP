//
//  ImageResizer.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 16/11/2022.
//

import Foundation
import UIKit

struct ImageRescaler: Equatable {
    let targetSize: CGSize
    
    init(
        targetSize: CGSize
    ) {
        self.targetSize = targetSize
    }
    
    func rescale(at url: URL) -> UIImage? {
        guard let image = UIImage(contentsOfFile: url.path) else { return nil }
        return callAsFunction(originalImage: image)
    }
    
    
    func callAsFunction(data: Data) -> UIImage? {
        guard let image = UIImage(data: data) else { return nil }
        return callAsFunction(originalImage: image)
    }
    
    func callAsFunction(originalImage: UIImage) -> UIImage {
        // Determine the scale factor that preserves aspect ratio
        let size: CGSize = originalImage.size
//        print("original size: \(size)")
        let widthRatio = targetSize.width / size.width
        let heightRatio = targetSize.height / size.height
        
        let scaleFactor = min(widthRatio, heightRatio)
        
        // Compute the new image size that preserves aspect ratio
        let scaledImageSize = CGSize(
            width: size.width * scaleFactor,
            height: size.height * scaleFactor
        )
        
        // Draw and return the resized UIImage
        let renderer = UIGraphicsImageRenderer(
            size: scaledImageSize
        )
        
        let scaledImage = renderer.image { _ in
            originalImage.draw(in: CGRect(
                origin: .zero,
                size: scaledImageSize
            ))
        }
        
//        print("rescaled's size: \(scaledImage.size)")
        return scaledImage
    }
}
