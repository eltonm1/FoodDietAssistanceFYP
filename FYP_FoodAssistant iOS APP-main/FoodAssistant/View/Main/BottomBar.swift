//
//  BottomBar.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 17/11/2022.
//

import SwiftUI

struct BottomBar: View {
    
    let screenSize: CGSize
    let onTabBarLeadingButtonTap: () -> Void
    let onTabBarCenterButtonTap: () -> Void
    let isTabBarCenterButtonMorphing: Bool
    let onTabBarTrailingButtonTap: () -> Void
    let normalizedCurrentTabOffset: CGFloat
    let tabScrollProgress: CGFloat
    let onCameraBottonBarLeadingLeadingButTap: () -> Void
    let onCameraBottonBarLeadingButTap: () -> Void
    let onCameraBottonBarTrailingButTap: () -> Void
    let onCameraBottonBarTrailingTrailingButTap: () -> Void
    let isPhotoCaptured: Bool
    let isScaleToFit: Bool
    let isFlashLightOn: Bool
    
    var body: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let size: CGSize = proxy.size
            VStack(spacing: 0) {
                CameraBottomBar(
                    onLeadingLeadingButTap: onCameraBottonBarLeadingLeadingButTap,
                    onLeadingButTap: onCameraBottonBarLeadingButTap,
                    onTrailingButTap: onCameraBottonBarTrailingButTap,
                    onTrailingTrailingButTap: onCameraBottonBarTrailingTrailingButTap,
                    isPhotoCaptured: isPhotoCaptured,
                    isScaleToFit: isScaleToFit,
                    isFlashLightOn: isFlashLightOn
                )
                .frame(
                    width: size.width,
                    height: 80 * tabScrollProgress
                )
                .opacity(tabScrollProgress)
                .animation(.easeOut, value: tabScrollProgress)
                TabBar(
                    screenSize: screenSize,
                    onLeadingButtonTap: onTabBarLeadingButtonTap,
                    onCenterButtonTap: onTabBarCenterButtonTap,
                    isCenterButtonMorphing: isTabBarCenterButtonMorphing,
                    onTrailingButtonTap: onTabBarTrailingButtonTap,
                    normalizedCurrentTabOffset: normalizedCurrentTabOffset,
                    tabScrollProgress: tabScrollProgress
                )
            }
            .background {
                Rectangle()
                    .fill(.ultraThinMaterial)
            }
        }
    }
    
}

struct BottomBar_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

