//
//  BoundingBox.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 8/11/2022.
//

import Foundation

struct BoundingBox: IdentifyEquateCodeHashable {
    let id = UUID()
    let classIndex: Int /** Index of the predicted class. */
    let score: Float /** Confidence score. */
    let rect: CGRect /** Normalized coordinates between 0 and 1. */
    
    init(
        classIndex: Int,
        score: Float,
        rect: CGRect
    ) {
        self.classIndex = classIndex
        self.score = score
        self.rect = rect
    }
}
