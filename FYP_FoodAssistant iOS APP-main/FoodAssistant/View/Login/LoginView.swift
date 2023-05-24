//
//  LoginView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 23/1/2023.
//

import SwiftUI

struct LoginView: View {
    let onButTap: () -> Void
    let gradient = LinearGradient(
        gradient: Gradient(stops: [
            .init(
                color: .adaptable(light: .white, dark: .black),
                location: 0.1
            ),
            .init(color: .clear, location: 1)
        ]),
        startPoint: .bottom,
        endPoint: .top
    )
    private let title = "FOOD ASSISTANT"
    private let subtitle = "Sign in to obtain nutrition information of thousand of products"
    private let butText = "Let's Get Started"
    var body: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let size: CGSize = proxy.size
            if size.width < size.height {
                PortraitView(
                    size: size,
                    gradient: gradient,
                    title: title,
                    subtitle: subtitle,
                    butText: butText,
                    onButTap: onButTap
                )
            } else {
                LandscapeView(
                    size: size,
                    gradient: gradient,
                    title: title,
                    subtitle: subtitle,
                    butText: butText,
                    onButTap: onButTap
                )
            }
        }
        .edgesIgnoringSafeArea(.all)
    }
}

private struct LandscapeView: View {
    @Environment(\.safeAreaInsets) var safeArea
    let size: CGSize
    let gradient: LinearGradient
    let title: String
    let subtitle: String
    let butText: String
    let onButTap: () -> Void
    var body: some View {
        HStack(alignment: .center, spacing: 24) {
            Image("banner")
                .resizable()
                .scaledToFill()
                .cornerRadius(40, style: .continuous)
                .frame(maxWidth: .infinity)
            VStack(spacing: 24) {
                Text(title)
                    .productFont(.bold, size: 38)
                    .foregroundColor(.primary)
                Text(subtitle)
                    .productFont(.regular, relativeTo: .title3)
                    .multilineTextAlignment(.center)
                    .foregroundColor(.primary.opacity(0.7))
                Button {
                    
                } label: {
                    Text(butText)
                        .productFont(.bold, relativeTo: .title2)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(.systemBlue)
                        .cornerRadius(20, style: .continuous)
                }
            }
            .frame(maxWidth: .infinity)
        }
        .padding()
        .frame(
            width: size.width - safeArea.trailing - safeArea.leading,
            height: size.height - safeArea.bottom * 2
        )
        .frame(
            maxWidth: .infinity,
            maxHeight: .infinity,
            alignment: .center
        )
        .background {
            Image("banner")
                .resizable()
                .padding(-64)
                .scaledToFill()
                .blur(radius: 64)
        }
    }
}

private struct PortraitView: View {
    @Environment(\.safeAreaInsets) var safeArea
    let size: CGSize
    let gradient: LinearGradient
    let title: String
    let subtitle: String
    let butText: String
    let onButTap: () -> Void
    var body: some View {
        Image("banner")
            .resizable()
            .scaledToFill()
            .frame(width: size.width, height: size.height)
            .overlay {
                ZStack(alignment: .bottom) {
                    Image("banner")
                        .resizable()
                        .scaledToFill()
                        .blur(radius: 20)
                        .padding(-20)
                        .frame(width: size.width, height: size.height)
                        .mask(gradient)
                    gradient
                    VStack(spacing: 24) {
                        Text(title)
                            .productFont(.bold, size: 38)
                            .foregroundColor(.primary)
                        Text(subtitle)
                            .productFont(.regular, relativeTo: .title3)
                            .multilineTextAlignment(.center)
                            .foregroundColor(.secondary)
                        Button {
                            
                        } label: {
                            Text(butText)
                                .productFont(.bold, relativeTo: .title2)
                                .foregroundColor(.white)
                                .padding()
                                .frame(maxWidth: .infinity)
                                .background(.systemBlue)
                                .cornerRadius(20, style: .continuous)
                        }
                    }
                    .padding(.bottom, 16 + safeArea.bottom)
                    .padding(.horizontal, 24)
                }
                .edgesIgnoringSafeArea(.all)
            }
    }
}

struct LoginView_Previews: PreviewProvider {
    static var previews: some View {
        LoginView {
            
        }
    }
}
