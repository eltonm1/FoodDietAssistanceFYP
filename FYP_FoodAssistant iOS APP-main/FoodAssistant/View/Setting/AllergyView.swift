//
//  AllergyView.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 6/1/2023.
//

import SwiftUI

struct AllergyView: View {
    @StateObject var vm = AllergyViewModel()
    @Namespace var ns
    let screenHeight: CGFloat
    var body: some View {
        List {
            SelectedAllergiesSession(vm: vm, ns: ns)
            AllAllergiesSession(vm: vm, ns: ns)
            Footer(screenHeight: screenHeight)
        }
            .productLargeNavigationBar()
            .navigationTitle("Allergy")
    }
}

fileprivate struct Footer: View {
    let screenHeight: CGFloat
    var body: some View {
        Section {
            
        } footer: {
            Rectangle()
                .opacity(0)
                .frame(height: screenHeight/8)
        }
    }
}

fileprivate struct SelectedAllergiesSession: View {
    @ObservedObject var vm: AllergyViewModel
    var ns: Namespace.ID
    var body: some View {
        Section {
            if vm.selectedAllergies.isEmpty {
                VStack {
                    Image("ingredients")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 200)
                    Text("You don't have any allergy now")
                        .productFont(.regular, relativeTo: .body)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .center)
            } else {
                ForEach(
                    vm.selectedAllergies, id: \.self
                ) { (i: Ingredient) in
                    HStack {
                        Button {
                            vm.removeAllergy(ingredient: i)
                        } label: {
                            Image(systemName: "minus.circle.fill")
                                .foregroundColor(.systemRed)
                        }
                        .hoverEffect()
                        Text(i.englishName)
                            .productFont(.regular, relativeTo: .body)
                    }
                        .matchedGeometryEffect(id: i, in: ns)
                }
            }
        } header: {
            Text("Your allergy ingredient")
                .productFont(.regular, relativeTo: .callout)
        }
    }
}

fileprivate struct AllAllergiesSession: View {
    @ObservedObject var vm: AllergyViewModel
    var ns: Namespace.ID
    var body: some View {
        Section {
            ForEach(
                vm.remainingAllergies, id: \.self
            ) { (i: Ingredient) in
                HStack {
                    Button {
                        vm.addAllergy(ingredient: i)
                    } label: {
                        Image(systemName: "plus.circle.fill")
                            .foregroundColor(.systemGreen)
                    }
                    .hoverEffect()
                    Text(i.englishName)
                        .productFont(.regular, relativeTo: .body)
                }
                .matchedGeometryEffect(id: i, in: ns)
            }
        } header: {
            Text("Included Allergies")
                .productFont(.regular, relativeTo: .callout)
        }
    }
}

struct AllergyView_Previews: PreviewProvider {
    @StateObject static var mvm = MainViewModel()
    static var previews: some View {
        GeometryReader { (proxy: GeometryProxy) in
            let height: CGFloat = proxy.size.height
            NavigationStack {
                AllergyView(screenHeight: height)
            }
        }
    }
}
