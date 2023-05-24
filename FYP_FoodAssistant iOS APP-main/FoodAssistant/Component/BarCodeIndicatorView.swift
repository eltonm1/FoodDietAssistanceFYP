//
//  BarCodeIndicatorView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 6/12/2022.
//

import SwiftUI

struct BarCodeIndicatorView: View {
    
    let barcode: String
    let width: CGFloat
    let height: CGFloat
    let offset: CGPoint
    
    @Binding var isAppeared: Bool
    
    var body: some View {
        ZStack(alignment: .topLeading) {
            Rectangle()
                .foregroundColor(.black)
                .opacity(isAppeared ? 0.4 : 0)
                .animation(.easeInOut, value: isAppeared)
            Rectangle()
                .frame(width: isAppeared ? width : 0, height: isAppeared ? height : 0)
                .cornerRadius(10)
                .blur(radius: 10)
                .blendMode(.destinationOut)
                .overlay {
                    RoundedRectangleCorners()
                        .stroke(lineWidth: 5)
                        .fill(.yellow)
                        .shadow(color: .systemYellow, radius: 10)
                }
                .overlay {
                    Text(barcode)
                        .productFont(.bold, relativeTo: .title2)
                        .foregroundColor(.white)
                        .shadow(color: .black, radius: 10)
                }
                .offset(isAppeared ? offset : offset + CGPoint(x: width/2, y: height/2))
                .animation(.easeInOut, value: isAppeared)
        }
        .compositingGroup()
    }
}

fileprivate extension CGPoint {
    static func +(lhs: Self, rhs: Self) -> CGPoint {
        .init(x: lhs.x + rhs.x, y: lhs.y + rhs.y)
    }
}
