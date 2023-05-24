//
//  AppState.swift
//  FoodAssistant
//
//  Created by Elton Miao on 28/12/2022.
//

import Foundation
import SwiftUI

class AppState {
    
    static let shared = AppState()
    
    // A helper class to fetch/store/retrieve data from server
    var dataService: APIService
    // A helper class to manage authentication matters
    var authService: AuthService
    
    private init() {
        let host = "20.205.60.6"//"192.168.1.9"//"127.0.0.1"
        self.authService = AuthService(
            scheme: "http",
            host: host,
            port: 8000
        )
        self.dataService = APIService(
            scheme: "http",
            host: host,
            port: 8000,
            authService: self.authService
        )
    }
}
