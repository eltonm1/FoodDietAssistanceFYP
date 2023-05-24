//
//  LoadingScreen.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 16/11/2022.
//

import SwiftUI

struct LaunchScreen: View {
    let screenSize: CGSize
    var body: some View {
        Image("Appicon")
            .resizable()
            .scaledToFit()
            .frame(width: screenSize.width/3, height: screenSize.height/3)
            .frame(width: screenSize.width, height: screenSize.height)
            .background(.white)
    }
}

struct LaunchScreen_Previews: PreviewProvider {
    @StateObject static var mvm = MainViewModel()
    static var previews: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let size: CGSize = proxy.size
            LaunchScreen(screenSize: size)
        }
            .environmentObject(mvm)
    }
}
