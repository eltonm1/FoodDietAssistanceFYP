//
//  SettingView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 17/11/2022.
//

import SwiftUI

struct SettingView: View {
    @EnvironmentObject var mvm: MainViewModel
    @StateObject var vm = SettingViewModel()
    let screenSize: CGSize
    var body: some View {
        NavigationStack {
            List {
                Section {
                    Row(
                        icon: "gear",
                        text: "General",
                        color: .systemGray,
                        destination: Text("hi")
                    )
                }
                Section {
                    Row(
                        icon: "fork.knife.circle",
                        text: "Food Allergy",
                        color: .systemRed,
                        destination: AllergyView(screenHeight: screenSize.height)
                    )
                    Row(
                        icon: "star.circle",
                        text: "Daily Calories Goal",
                        color: .systemOrange,
                        destination: Text("hi")
                    )
//                    Row(
//                        icon: "list.bullet.circle.fill",
//                        text: "Inventory",
//                        color: .systemBlue,
//                        destination: InventoryView(dataSource: mvm.foodDataService)
//                    )
                } footer: {
                    Rectangle()
                        .frame(height: screenSize.height/8)
                        .opacity(0)
                }
            }
            .edgesIgnoringSafeArea(.bottom)
            .navigationTitle("Setting")
            .nativeSearchBar(text: $vm.searchedSetting, placeHolder: "Search Setting")
        }
        .productLargeNavigationBar()
        .frame(width: screenSize.width, height: screenSize.height)
    }
}

extension SettingView {
    struct Row<V: View>: View {
        let icon: String
        let text: String
        let color: Color
        let destination: V
        var body: some View {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(.white)
                    .padding(4)
                    .background {
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .foregroundColor(color)
                    }
                Text(text)
                    .productFont(.regular, relativeTo: .body)
                    .foregroundColor(.primary)
                Spacer()
                Image(systemName: "chevron.right")
                    .foregroundColor(.secondary)
            }
            .onPress(navigateTo: destination)
        }
    }
}

struct SettingView_Previews: PreviewProvider {
    @StateObject static var mvm = MainViewModel()
    static var previews: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let size: CGSize = proxy.size
            SettingView(screenSize: size)
        }
        .environmentObject(mvm)
    }
}
