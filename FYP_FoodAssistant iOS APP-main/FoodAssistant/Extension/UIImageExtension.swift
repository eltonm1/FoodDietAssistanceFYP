//
//  UIImageExtension.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 16/11/2022.
//

import UIKit

extension UIImage {
    enum Position {                                         // the position of the image in the canvas
        case top
        case bottom
        case left
        case right
        case middle
        case topLeft
        case topRight
        case bottomLeft
        case bottomRight
    }
    
    func drawOnCanvas(withCanvasSize canvasSize: CGSize,
                      andCanvasColor canvasColor: UIColor,
                      atPosition position: Position) -> UIImage {
        let rect = CGRect(origin: .zero, size: canvasSize)
        UIGraphicsBeginImageContextWithOptions(rect.size, false, 0.0)
        canvasColor.setFill()                               // fill the entire image
        UIRectFill(rect)
        
        var x = 0.0                                         // the position of the image in the canvas
        var y = 0.0
        switch position {                                   // look at the position specified
        case .top:
            x = (canvasSize.width - self.size.width) / 2    // position horizontally in the middle
        case .bottom:
            x = (canvasSize.width - self.size.width) / 2    // position horizontally in the middle
            y = canvasSize.height - self.size.height        // and vertically at the bottom
        case .left:
            y = (canvasSize.height - self.size.height) / 2  // position vertically in the middle
        case .right:
            x = canvasSize.width - self.size.width          // position horizontally at the right
            y = (canvasSize.height - self.size.height) / 2  // and vertically in the middle
        case .middle:
            x = (canvasSize.width - self.size.width) / 2    // position horizontally in the middle
            y = (canvasSize.height - self.size.height) / 2  // and vertically in the middle
        case .topLeft:
            x = 0.0                                         // just dummy
        case .topRight:
            x = canvasSize.width - self.size.width          // position horizontally at the right
        case .bottomLeft:
            y = canvasSize.height - self.size.height        // position vertically at the bottom
        case .bottomRight:
            x = canvasSize.width - self.size.width          // position horizontally at the right
            y = canvasSize.height - self.size.height        // and vertically at the bottom
        }
        // calculate a Rect the size of the image to draw, centered in the canvas rect
        let positionedImageRect = CGRect(x: x,              // position the image in the canvas
                                         y: y,
                                         width: self.size.width,
                                         height: self.size.height)
        let context = UIGraphicsGetCurrentContext()         // get a drawing context
        // "cut" a transparent rectanlge in the middle of the "canvas" image
        context?.clear(positionedImageRect)
        self.draw(in: positionedImageRect)                  // draw the image into that rect
        let image = UIGraphicsGetImageFromCurrentImageContext() // get the new "image in the canvas image"
        UIGraphicsEndImageContext()
        return image!                                       // and return it
    }
}
