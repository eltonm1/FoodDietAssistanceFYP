//
//  CameraBottomBar.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 16/11/2022.
//

import SwiftUI

struct CameraBottomBar: View {
    
    @StateObject var vm = CameraBottomBarViewModel()
    
    let onLeadingLeadingButTap: () -> Void
    let onLeadingButTap: () -> Void
    let onTrailingButTap: () -> Void
    let onTrailingTrailingButTap: () -> Void
    let isPhotoCaptured: Bool
    let isScaleToFit: Bool
    let isFlashLightOn: Bool
    
    var body: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let size: CGSize = proxy.size
            HStack {
                Group {
                    LeadingLeadingButton(
                        size: size,
                        onTap: onLeadingLeadingButTap
                    )
                    LeadingButton(
                        size: size,
                        onTap: onLeadingButTap
                    )
                    CenterButton(size: size)
                    TrailingButton(
                        size: size,
                        isPhotoCaptured: isPhotoCaptured,
                        isScaleToFit: isScaleToFit,
                        onTap: onTrailingButTap
                    )
                    TrailingTrailingButton(
                        size: size,
                        isFlashLightOn: isFlashLightOn,
                        onTap: onTrailingTrailingButTap
                    )
                }
                .foregroundColor(.primary)
                .frame(maxWidth: .infinity, alignment: .center)
                .padding(.vertical)
            }
            .frame(width: size.width, height: size.height)
        }
        .onReceive(
            Timer
                .publish(
                    every: 0.01,
                    on: .main,
                    in: .default
                )
                .autoconnect()
        ) { _ in vm.onTimerUpdate() }
        .overlay(alignment: .top) {
            ZStack {
                if vm.showFlashLightLabel {
                    FlashLightLabel(
                        isFlashLightOn: isFlashLightOn
                    )
                }
            }
            .alignmentGuide(.top) { $0[.bottom] + 32 }
        }
        .onChange(of: isFlashLightOn) { _ in vm.onFlashLightModeToggle() }
    }
}

fileprivate struct FlashLightLabel: View {
    let isFlashLightOn: Bool
    var body: some View {
        HStack {
            Image(
                systemName: isFlashLightOn ?
                "bolt.circle" : "bolt.slash.circle"
            )
            .font(.body)
            .rotationEffect(isFlashLightOn ? .pi : .zero)
            .animation(.spring(), value: isFlashLightOn)
            .transition(.opacity)
            Text("Flash Light is \(isFlashLightOn ? "On" : "Off") now")
                .productFont(.bold, relativeTo: .body)
        }
        .foregroundColor(isFlashLightOn ? .black : .white)
        .padding(12)
        .padding(.horizontal)
        .background(
            BlurMaterialView(
                isFlashLightOn ? .systemThinMaterialLight : .systemThinMaterialDark
            )
        )
        .clipShape(Capsule())
        .shadow(radius: 20)
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }
}

fileprivate struct LeadingLeadingButton: View {
    let size: CGSize
    let onTap: () -> Void
    var body: some View {
        SFButton("captions.bubble", action: onTap)
            .scaledToFit()
            .padding(12)
            .frame(
                width: max(0, size.height - 32),
                height: max(0, size.height - 32)
            )
            .background {
                Circle()
                    .fill(.regularMaterial)
            }
            .hoverEffect()
    }
}

fileprivate struct LeadingButton: View {
    let size: CGSize
    let onTap: () -> Void
    var body: some View {
        SFButton("photo", action: onTap)
            .scaledToFit()
            .padding(12)
            .frame(
                width: max(0, size.height - 32),
                height: max(0, size.height - 32)
            )
            .background {
                Circle()
                    .fill(.regularMaterial)
            }
            .hoverEffect()
    }
}

fileprivate struct CenterButton: View {
    let size: CGSize
    var body: some View {
        SFButton("photo")
            .scaledToFit()
            .frame(
                width: max(0, size.height - 32),
                height: max(0, size.height - 32)
            )
            .opacity(0)
    }
}

fileprivate struct TrailingButton: View {
    let size: CGSize
    let isPhotoCaptured: Bool
    let isScaleToFit: Bool
    let onTap: () -> Void
    var body: some View {
        SFButton(
            !isPhotoCaptured ? "arrow.triangle.2.circlepath.camera" :
                (isScaleToFit ?  "arrow.down.right.and.arrow.up.left.circle" : "arrow.up.backward.and.arrow.down.forward.circle"),
            action: onTap
        )
        .scaledToFit()
        .rotationEffect(
            isPhotoCaptured ? (isScaleToFit ? .pi : .zero) : .zero
        )
        .animation(.spring(), value: isScaleToFit)
        .padding(12)
        .frame(
            width: max(0, size.height - 32),
            height: max(0, size.height - 32)
        )
        .background {
            Circle()
                .fill(.regularMaterial)
        }
        .hoverEffect()
    }
}

fileprivate struct TrailingTrailingButton: View {
    let size: CGSize
    let isFlashLightOn: Bool
    let onTap: () -> Void
    var body: some View {
        SFButton(
            isFlashLightOn ? "bolt.slash.circle" : "bolt.circle",
            action: onTap
        )
        .scaledToFit()
        .padding(12)
        .frame(
            width: max(0, size.height - 32),
            height: max(0, size.height - 32)
        )
        .background {
            Circle()
                .fill(.regularMaterial)
        }
        .hoverEffect()
    }
}

struct CameraBottomBar_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
