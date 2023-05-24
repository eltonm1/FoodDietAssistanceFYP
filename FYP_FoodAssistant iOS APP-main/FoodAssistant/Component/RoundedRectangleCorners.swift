//
//  RoundedRectangleCorners.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 4/12/2022.
//

import SwiftUI

struct RoundedRectangleCorners: Shape {
    
    let delta: CGFloat = 30
    
    func path(in rect: CGRect) -> Path {
        Path { (path: inout Path) in
            path.move(to: CGPoint(x: rect.minX, y: rect.minY+delta))
            path.addCurve(
                to: CGPoint(x: rect.minX+delta, y: rect.minY),
                control1: CGPoint(x: rect.minX, y: rect.minY),
                control2: CGPoint(x: rect.minX, y: rect.minY)
            )
            path.move(to: CGPoint(x: rect.maxX, y: rect.minY+delta))
            path.addCurve(
                to: CGPoint(x: rect.maxX-delta, y: rect.minY),
                control1: CGPoint(x: rect.maxX, y: rect.minY),
                control2: CGPoint(x: rect.maxX, y: rect.minY)
            )
            path.move(to: CGPoint(x: rect.minX, y: rect.maxY-delta))
            path.addCurve(
                to: CGPoint(x: rect.minX+delta, y: rect.maxY),
                control1: CGPoint(x: rect.minX, y: rect.maxY),
                control2: CGPoint(x: rect.minX, y: rect.maxY)
            )
            path.move(to: CGPoint(x: rect.maxX, y: rect.maxY-delta))
            path.addCurve(
                to: CGPoint(x: rect.maxX-delta, y: rect.maxY),
                control1: CGPoint(x: rect.maxX, y: rect.maxY),
                control2: CGPoint(x: rect.maxX, y: rect.maxY)
            )
        }
    }
}
