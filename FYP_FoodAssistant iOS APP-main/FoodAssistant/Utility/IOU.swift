//
//  IOU.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 16/11/2022.
//

import Foundation

struct IOU {
    let a: CGRect
    let b: CGRect
    
    init(
        _ a: CGRect,
        _ b: CGRect
    ) {
        self.a = a
        self.b = b
    }
    
    func callAsFunction() -> Float {
      let areaA = a.width * a.height
      if areaA <= 0 { return 0 }

      let areaB = b.width * b.height
      if areaB <= 0 { return 0 }

      let intersectionMinX = max(a.minX, b.minX)
      let intersectionMinY = max(a.minY, b.minY)
      let intersectionMaxX = min(a.maxX, b.maxX)
      let intersectionMaxY = min(a.maxY, b.maxY)
      let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
                             max(intersectionMaxX - intersectionMinX, 0)
      return Float(intersectionArea / (areaA + areaB - intersectionArea))
    }
}
