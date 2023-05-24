//
//  BoundingBoxView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 16/11/2022.
//

import SwiftUI
import SwiftUIX

struct BoundingBoxView: View {
    
    let bboxes: [BoundingBox]
    let size: CGSize
    let offset: CGFloat
    let ratio: CGFloat
    let rescaledSize: CGSize
    
    init(
        boundingBoxes: [BoundingBox],
        size: CGSize,
        rescaledSize: CGSize
    ) {
        self.bboxes = boundingBoxes
        self.size = size
        self.rescaledSize = rescaledSize
        
        self.offset = (416 - min(rescaledSize.width, rescaledSize.height))/2
        self.ratio = min(size.width, size.height)/min(rescaledSize.width, rescaledSize.height)
    }
    
    var body: some View {
        ForEach(bboxes) { (bbox: BoundingBox) in
            let x: CGFloat = (bbox.rect.minX - bbox.rect.width/2 - self.offset) * ratio
            let y: CGFloat = (bbox.rect.minY - bbox.rect.height/2) * ratio
            Text("\(bbox.score)")
                .productFont(.bold, relativeTo: .caption)
                .frame("\(bbox.score)".productAttributeSize(.bold, relativeTo: .subheadline))
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
                .offset(x: x - 20, y: y - 20)
            Rectangle()
                .stroke(Color.red, lineWidth: 3.0)
                .frame(width: bbox.rect.width * ratio, height: bbox.rect.height * ratio)
                .offset(x: x, y: y)
        }
    }
}
