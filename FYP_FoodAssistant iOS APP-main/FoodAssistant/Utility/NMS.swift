//
//  NMS.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 16/11/2022.
//

import Foundation

struct NMS {
    let bboxes: [BoundingBox]
    let iouThreshold: Float
    let maxBoxes: Int
    
    /**
     Removes bounding boxes that overlap too much with other boxes that have
     a higher score.
     Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc
     - Note: This version of NMS ignores the class of the bounding boxes. Since it
     selects the bounding boxes in a greedy fashion, if a certain class has many
     boxes that are selected, then it is possible none of the boxes of the other
     classes get selected.
     - Parameters:
     - boundingBoxes: an array of bounding boxes and their scores
     - indices: which predictions to look at
     - iouThreshold: used to decide whether boxes overlap too much
     - maxBoxes: the maximum number of boxes that will be selected
     - Returns: the array indices of the selected bounding boxes
     */
    init(
        bboxes: [BoundingBox],
        iouThreshold: Float,
        maxBoxes: Int
    ) {
        self.bboxes = bboxes
        self.iouThreshold = iouThreshold
        self.maxBoxes = maxBoxes
    }
    
    func callAsFunction() -> [Int] {
        let indices: [Int] = Array(bboxes.indices)
        // Sort the boxes based on their confidence scores, from high to low.
        let sortedIndices: [Int] = indices.sorted { bboxes[$0].score > bboxes[$1].score }
        
        var selected: [Int] = []
        
        // Loop through the bounding boxes, from highest score to lowest score,
        // and determine whether or not to keep each box.
        for i in 0..<sortedIndices.count {
            if selected.count >= maxBoxes { break }
            
            var shouldSelect = true
            let boxA = bboxes[sortedIndices[i]]
            
            // Does the current box overlap one of the selected boxes more than the
            // given threshold amount? Then it's too similar, so don't keep it.
            for j in 0..<selected.count {
                let boxB = bboxes[selected[j]]
                if IOU(boxA.rect, boxB.rect)() > iouThreshold {
                    shouldSelect = false
                    break
                }
            }
            
            // This bounding box did not overlap too much with any previously selected
            // bounding box, so we'll keep it.
            if shouldSelect {
                selected.append(sortedIndices[i])
            }
        }
        
        return selected
    }
}
