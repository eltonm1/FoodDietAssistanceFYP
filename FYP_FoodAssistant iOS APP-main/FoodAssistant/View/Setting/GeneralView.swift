//
//  GeneralView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 22/12/2022.
//

import SwiftUI
import Introspect

extension SettingView {
    struct GeneralView: View {
        @ObservedObject var vm: SettingViewModel
        var body: some View {
            List {
                Section {
                    StartPageRow(vm: vm)
                } header: {
                    Text("Start Page")
                        .foregroundColor(.secondary)
                        .productFont(.regular, relativeTo: .body)
                } footer: {
                    Text("Choose the page you want to start with")
                        .foregroundColor(.secondary)
                        .productFont(.regular, relativeTo: .body)
                }
            }
            .productLargeNavigationBar()
            .navigationTitle("General")
        }
    }
}

fileprivate struct StartPageRow: View {
    @ObservedObject var vm: SettingViewModel
    var body: some View {
        HStack {
            But(
                selectedStartPage: $vm.startPage,
                title: "Catagory View",
                image: "list.dash",
                color: .systemRed,
                startPage: .catagory
            )
            But(
                selectedStartPage: $vm.startPage,
                title: "Camera View",
                image: "circle.inset.filled",
                color: .systemOrange,
                startPage: .camera
            )
            But(
                selectedStartPage: $vm.startPage,
                title: "Setting View",
                image: "gear.circle",
                color: .systemBlue,
                startPage: .setting
            )
        }
        .introspectTableViewCell { (cell: UITableViewCell) in
            // make the cell unselectable and unclicable
            cell.selectionStyle = .none
            cell.isUserInteractionEnabled = false
        }
    }

    fileprivate struct But: View {
        @Binding var selectedStartPage: SettingViewModel.StartPage
        let title: String
        let image: String
        let color: Color
        let startPage: SettingViewModel.StartPage
        
        var body: some View {
            let isSelected: Bool = selectedStartPage == startPage
            Button {
                withAnimation(.spring()) {
                    selectedStartPage = startPage
                }
            } label: {
                VStack(spacing: 16) {
                    Image(systemName: image)
                        .resizable()
                        .scaledToFit()
                        .foregroundColor(color)
                        .frame(width: 50)
                    Text(title)
                        .foregroundColor(color)
                        .productFont(.regular, relativeTo: .body)
                        .multilineTextAlignment(.center)
                }
                .padding()
                .frame(height: 150)
                .frame(maxWidth: .infinity)
                .background {
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .foregroundColor(color)
                        .opacity(isSelected ? 0.1 : 0)
                }
            }
        }
    }
}

struct GeneralView_Previews: PreviewProvider {
    @StateObject static var vm = SettingViewModel()
    static var previews: some View {
        NavigationStack {
            SettingView.GeneralView(vm: vm)
        }
    }
}
