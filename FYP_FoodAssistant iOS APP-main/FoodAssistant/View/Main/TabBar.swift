//
//  TabBar.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 17/11/2022.
//

import SwiftUI

struct TabBar: View {
    
    @Environment(\.safeAreaInsets) var safeArea
    let screenSize: CGSize
    let onLeadingButtonTap: () -> Void
    let onCenterButtonTap: () -> Void
    let isCenterButtonMorphing: Bool
    let onTrailingButtonTap: () -> Void
    let normalizedCurrentTabOffset: CGFloat
    let tabScrollProgress: CGFloat
    
    var body: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let size: CGSize = proxy.size
            HStack {
                LeadingButton(
                    screenSize: screenSize,
                    onTap: onLeadingButtonTap,
                    normalizedCurrentTabOffset: normalizedCurrentTabOffset
                )
                CenterButton(
                    screenSize: screenSize,
                    onTap: onCenterButtonTap,
                    normalizedCurrentTabOffset: normalizedCurrentTabOffset,
                    tabScrollProgress: tabScrollProgress,
                    isMorphing: isCenterButtonMorphing
                )
                TrailingButton(
                    screenSize: screenSize,
                    onTap: onTrailingButtonTap,
                    normalizedCurrentTabOffset: normalizedCurrentTabOffset
                )
            }
            .padding(.bottom, safeArea.bottom)
            .frame(width: size.width, height: size.height)
        }
    }
    
    fileprivate struct LeadingButton: View {
        @Environment(\.safeAreaInsets) var safeArea
        let screenSize: CGSize
        let onTap: () -> Void
        let normalizedCurrentTabOffset: CGFloat
        var body: some View {
            GeometryReader { (proxy: GeometryProxy) in
                let height: CGFloat = proxy.size.height
                let size: CGFloat = min(50, height)
                let isPortrait: Bool = screenSize.width < screenSize.height
                ZStack {
                    Color.blue
                    Color.primary
                        .opacity(min(1, normalizedCurrentTabOffset))
                }
                .mask {
                    SFButton("list.dash")
                        .scaledToFit()
                        .frame(width: size, height: size)
                }
                .frame(width: size, height: size - (isPortrait ? 0 : 10))
                .hoverEffect()
                .padding(.top, isPortrait ? 0 : 10)
                .onTapGesture(perform: onTap)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
    
    fileprivate struct CenterButton: View {
        @Environment(\.safeAreaInsets) var safeArea
        let screenSize: CGSize
        let onTap: () -> Void
        let normalizedCurrentTabOffset: CGFloat
        let tabScrollProgress: CGFloat
        let isMorphing: Bool
        var body: some View {
            GeometryReader { (proxy: GeometryProxy) in
                let height: CGFloat = proxy.size.height
                let size: CGFloat = min(50, height)
                let isPortrait: Bool = screenSize.width < screenSize.height
                ZStack {
                    Color.blue
                    Color.primary
                        .opacity(
                            normalizedCurrentTabOffset <= 1 ?
                            1 - normalizedCurrentTabOffset : normalizedCurrentTabOffset - 1
                        )
                }
                .mask {
                    MorphingView(
                        isTapped: Binding<Bool>(
                            get: { isMorphing },
                            set: { _ in }
                        )
                    )
                }
                .frame(width: size, height: size - (isPortrait ? 0 : 10))
                .padding(.top, isPortrait ? 0 : 10)
                .scaleEffect( 1 + tabScrollProgress, anchor: .bottom)
                .animation(.easeOut, value: tabScrollProgress)
                .onTapGesture(perform: onTap)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
    
    fileprivate struct TrailingButton: View {
        @Environment(\.safeAreaInsets) var safeArea
        let screenSize: CGSize
        let onTap: () -> Void
        let normalizedCurrentTabOffset: CGFloat
        var body: some View {
            GeometryReader { (proxy: GeometryProxy) in
                let height: CGFloat = proxy.size.height
                let size: CGFloat = min(50, height)
                let isPortrait: Bool = screenSize.width < screenSize.height
                ZStack {
                    Color.blue
                    Color.primary
                        .opacity(2 - normalizedCurrentTabOffset)
                }
                .mask {
                    SFButton("folder")
                        .scaledToFit()
                }
                .frame(width: size, height: size - (isPortrait ? 0 : 10))
                .hoverEffect()
                .padding(.top, isPortrait ? 0 : 10)
                .onTapGesture(perform: onTap)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }
}


struct TabBar_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
