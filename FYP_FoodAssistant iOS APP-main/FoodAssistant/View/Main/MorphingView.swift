//
//  MorphingView.swift
//  Morphing
//
//  Created by Balaji on 18/09/22.
//

import SwiftUI

struct MorphingView: View {
    
    @Binding var isTapped: Bool
    
    @State var currentImage: String = "record.circle.fill"
    @State var blurRadius: CGFloat = 0
    @State var animateMorph: Bool = false
    
    var body: some View {
        // MARK: Image Morph is Simple
        // Simply Mask the Canvas Shape as Image Mask
        GeometryReader { (proxy: GeometryProxy) in
            let size: CGSize = proxy.size
            Color.primary
                .frame(width: size.width, height: size.height)
                .clipped()
                .mask {
                    // MARK: Morphing Shapes With the Help of Canvas and Filters
                    Canvas{context,size in
                        // MARK: Morphing Filters
                        // For More Morph Shape Link
                        // MARK: For More Morph Shape Link Change This
                        context.addFilter(.alphaThreshold(min: 0.35))
                        // MARK: This Value Plays Major Role in the Morphing Animation
                        // MARK: For Reverse Animation
                        // Until 20 -> It will be like 0-1
                        // After 20 Till 40 -> It will be like 1-0
                        context.addFilter(.blur(radius: blurRadius >= 20 ? 20 - (blurRadius - 20) : blurRadius))
                        
                        // MARK: Draw Inside Layer
                        context.drawLayer { ctx in
                            if let resolvedImage = context.resolveSymbol(id: 1) {
                                ctx.draw(resolvedImage, at: CGPoint(x: size.width / 2, y: size.height / 2), anchor: .center)
                            }
                        }
                    } symbols: {
                        // MARK: Giving Images With ID
                        ResolvedImage(currentImage: $currentImage)
                            .tag(1)
                    }
                    // MARK: Animations will not Work in the Canvas
                    // We can use Timeline View For those Animations
                    // But here I'm going to simply Use Timer to Acheive the Same Effect
                    
                    // The Timer Value is Animation Speed
                    // You can Change this for your Own
                    // EG: For Optimal Speed Use = 0.007
                    .onReceive(
                        Timer
                            .publish(every: 0.01, on: .main, in: .common)
                            .autoconnect()
                    ) { _ in
                        if animateMorph {
                            if blurRadius <= 40{
                                blurRadius += 0.5
                                
                                if blurRadius.rounded() == 20 {
                                    // MARK: Change Of Next Image Goes Here
                                    currentImage = isTapped ? "play.circle.fill" : "record.circle.fill"
                                }
                            }
                            
                            if blurRadius.rounded() == 40{
                                // MARK: End Animation And Reset the Blur Radius to Zero
                                animateMorph = false
                                blurRadius = 0
                            }
                        }
                    }
                }
        }
        .onChange(of: isTapped) { _ in
            animateMorph = true
        }
    }
}

struct ResolvedImage: View{
    @Binding var currentImage: String
    var body: some View{
        Image(systemName: currentImage)
//            .font(.system(size: 200))
            .resizable()
            .scaledToFit()
            .animation(.interactiveSpring(response: 0.7, dampingFraction: 0.8, blendDuration: 0.8), value: currentImage)
//            .frame(width: 300, height: 300)
    }
}

struct MorphingView_Previews: PreviewProvider {
    @State static var isTapped = false
    static var previews: some View {
        MorphingView(isTapped: $isTapped)
            .onTapGesture {
                isTapped.toggle()
            }
    }
}
