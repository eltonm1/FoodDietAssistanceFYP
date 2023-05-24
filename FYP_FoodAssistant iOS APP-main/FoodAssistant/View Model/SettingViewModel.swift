//
//  SettingViewModel.swift
//  FoodAssistant
//
//  Created by Choi Wai Lap on 22/12/2022.
//

import Foundation

class SettingViewModel: ObservableObject {
    
    enum StartPage: CaseIterable {
        case catagory, camera, setting
    }
    
    @Published var searchedSetting = ""
    
    // General Page
    @Published var startPage: StartPage = .camera
}
