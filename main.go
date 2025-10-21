package main

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func longestMonotonicSubarray(nums []int) int {
	if len(nums) == 1 {
		return 1
	}
	var maxLength int = 1
	var upLength int = 1
	var downLength int = 1
	for i := 1; i < len(nums); i++ {
		if nums[i] > nums[i-1] {
			upLength++
			downLength = 1
		}
		if nums[i] < nums[i-1] {
			downLength++
			upLength = 1
		}
		if nums[i] == nums[i-1] {
			downLength = 1
			upLength = 1
		}
		if downLength > maxLength {
			maxLength = downLength
		}
		if upLength > maxLength {
			maxLength = upLength
		}
	}
	return maxLength
}

func singleNumber(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	var xorElement = nums[0]
	for i := 1; i < len(nums); i++ {
		xorElement = xorElement ^ nums[i]
	}
	return xorElement
}

func maxSubarraySumCircular(nums []int) int {
	var maxSum = nums[0]
	var minSum = nums[0]
	var tempMax = 0
	var tempMin = 0
	var totalSum = 0

	for i := 0; i < len(nums); i++ {
		tempMax = max(tempMax, 0) + nums[i]
		maxSum = max(tempMax, maxSum)

		tempMin = min(tempMin, 0) + nums[i]
		minSum = min(tempMin, minSum)

		totalSum += nums[i]
	}

	if minSum == totalSum {
		return maxSum
	} else {
		return max(maxSum, totalSum-minSum)
	}
}

func candy(ratings []int) int {
	var n = len(ratings)
	var totalCandies = n
	for i := 1; i < n; {
		if ratings[i] == ratings[i-1] {
			i++
			continue
		}
		var currentPeek = 0
		for i < n && ratings[i] > ratings[i-1] {
			currentPeek++
			totalCandies += currentPeek
			i++
		}
		if i == n {
			return totalCandies
		}
		var currentValley = 0
		for i < n && ratings[i] < ratings[i-1] {
			currentValley++
			totalCandies += currentValley
			i++
		}
		totalCandies -= min(currentValley, currentPeek)
	}
	return totalCandies
}

func coinChange(coins []int, amount int) int {
	var dp = make([]int, amount+1)
	for i := 0; i < len(dp); i++ {
		dp[i] = amount + 1
	}
	dp[0] = 0
	for i := 1; i < len(dp); i++ {
		for j := 0; j < len(coins); j++ {
			if i-coins[j] >= 0 {
				dp[i] = min(dp[i], dp[i-coins[j]]+1)
			}
		}
	}
	return dp[amount]
}

var happenedNumber []int

func getTotalSquare(number int) int {
	var res int = 0
	if number == 1 {
		return 1
	}
	for number > 0 {
		res += (number % 10) * (number % 10)
		number /= 10
	}
	if slices.Contains(happenedNumber, res) {
		return -1
	}
	happenedNumber = append(happenedNumber, res)
	return getTotalSquare(res)
}

func isHappy(n int) bool {
	return getTotalSquare(n) == 1
}

func maxAscendingSum(nums []int) int {
	var maxSum int = nums[0]
	var tempSum int = nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i] > nums[i-1] {
			tempSum += nums[i]
		} else {
			if tempSum > maxSum {
				maxSum = tempSum
			}
			tempSum = nums[i]
		}
		if tempSum > maxSum {
			maxSum = tempSum
		}
	}
	return maxSum
}
func areAlmostEqual(s1 string, s2 string) bool {
	var diffCount = 0
	var diffColls1 = make([]byte, 0)
	var diffColls2 = make([]byte, 0)
	for i := 0; i < len(s1); i++ {
		if s1[i] != s2[i] {
			diffCount++
			diffColls2 = append(diffColls2, s2[i])
			diffColls1 = append(diffColls1, s1[i])
		}
		if diffCount > 2 {
			return false
		}
	}
	if diffCount == 2 {
		return diffColls1[0] == diffColls2[1] && diffColls1[1] == diffColls2[0]
	}
	if diffCount != 0 {
		return false
	} else {
		return true
	}
}

func tupleSameProduct(nums []int) int {
	var freqMap = make(map[int]int)
	for i := 0; i < len(nums)-1; i++ {
		for j := i + 1; j < len(nums); j++ {
			if freqMap[nums[i]*nums[j]] == 0 {
				freqMap[nums[i]*nums[j]] = 1
			} else {
				freqMap[nums[i]*nums[j]]++
			}
		}
	}
	var count int = 0
	for _, value := range freqMap {
		if value > 1 {
			count += 4 * value * (value - 1)
		}
	}
	return count
}

func queryResults(limit int, queries [][]int) []int {
	var result = make([]int, len(queries))
	var freqMap = make(map[int]int)
	var coloredBall = make(map[int]int)
	var diffColorCount = 0
	for i := 0; i < len(queries); i++ {
		if coloredBall[queries[i][0]] == 0 {
			if freqMap[queries[i][1]] == 0 {
				diffColorCount++
			}
		} else {
			var oldColor = coloredBall[queries[i][0]]
			var newColor = queries[i][1]
			freqMap[oldColor]--
			if freqMap[oldColor] == 0 {
				diffColorCount--
			}
			if freqMap[newColor] == 0 {
				diffColorCount++
			}
		}
		coloredBall[queries[i][0]] = queries[i][1]
		freqMap[queries[i][1]]++
		result[i] = diffColorCount
	}
	return result
}

type NumberContainers struct {
	numbers map[int]int
	index   map[int][]int
}

func NumberContainersConstructor() NumberContainers {
	return NumberContainers{
		numbers: make(map[int]int),
		index:   make(map[int][]int),
	}
}

func (this *NumberContainers) Change(index int, number int) {
	var oldValue = this.numbers[index]
	this.numbers[index] = number
	if oldValue != 0 {
		this.index[oldValue] = DeleteSorted(this.index[oldValue], index)
	}
	this.index[number] = InsertSorted(this.index[number], index)
}

func (this *NumberContainers) Find(number int) int {
	if this.index[number] != nil {
		return -1
	} else {
		return this.index[number][0]
	}
}

func InsertSorted(arr []int, val int) []int {
	idx := sort.SearchInts(arr, val)
	arr = append(arr, 0)
	copy(arr[idx+1:], arr[idx:])
	arr[idx] = val
	return arr
}

func DeleteSorted(arr []int, val int) []int {
	idx := sort.SearchInts(arr, val)
	if idx < len(arr) && arr[idx] == val {
		arr = append(arr[:idx], arr[idx+1:]...)
	}
	return arr
}

func countBadPairs(nums []int) int64 {
	var notBadPairCount = 0
	var mapper = make(map[int]int)
	for i := 0; i < len(nums); i++ {
		mapper[nums[i]-i]++
		notBadPairCount += mapper[nums[i]-i]
	}

	return int64(len(nums)*(len(nums)-1)/2 - notBadPairCount)
}

type stack[T any] struct {
	Push   func(T)
	Pop    func() T
	Length func() int
}

func Stack[T any]() stack[T] {
	slice := make([]T, 0)
	return stack[T]{
		Push: func(i T) {
			slice = append(slice, i)
		},
		Pop: func() T {
			res := slice[len(slice)-1]
			slice = slice[:len(slice)-1]
			return res
		},
		Length: func() int {
			return len(slice)
		},
	}
}
func solve(index int, digits string, comb []string, ans *[]string, temp string) {
	if index == len(digits) {
		*ans = append(*ans, temp)
		return
	}

	for _, ch := range comb[digits[index]-'0'] {
		solve(index+1, digits, comb, ans, temp+string(ch))
	}
}

func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return []string{}
	}
	comb := []string{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}
	var ans []string
	solve(0, digits, comb, &ans, "")
	return ans
}
func zeroFilledSubarray(nums []int) int64 {
	result := 0
	isValid := false
	startIndex := -1
	endIndex := -1
	for i := 0; i < len(nums); i++ {
		if nums[i] == 0 {
			if isValid {
				endIndex = i
			} else {
				isValid = true
				startIndex = i
				endIndex = i
			}
			if i == len(nums)-1 {
				n := endIndex - startIndex + 1
				result += n + (n-1)*n/2
			}
		} else if nums[i] != 0 && isValid {
			n := endIndex - startIndex + 1
			result += n + (n-1)*n/2
			isValid = false
		}
	}
	return int64(result)
}

func repeatedCharacter(s string) byte {
	vector := 0

	for _, c := range s {
		if vector&(1<<(c-'a')) > 0 {
			return byte(c)
		}
		vector |= 1 << (c - 'a')
	}
	return ' '
}

func equalPairs(grid [][]int) int {
	var result int = 0
	hashedRow := make(map[string]int)
	for i := 0; i < len(grid); i++ {
		var newRow string = ""
		for j := 0; j < len(grid[i]); j++ {
			newRow += string(grid[i][j] * 10)
		}
		hashedRow[newRow]++
	}
	for i := 0; i < len(grid[0]); i++ {
		var newRow string = ""
		for j := 0; j < len(grid); j++ {
			newRow += string(grid[j][i] * 10)
		}
		if hashedRow[newRow] > 0 {
			result += hashedRow[newRow]
		}
	}
	return result
}

func getHappyString(n int, k int) string {
	stringStack := []string{""}
	index := 0
	for len(stringStack) > 0 {
		current := stringStack[len(stringStack)-1]
		stringStack = stringStack[:len(stringStack)-1]
		if len(current) == n {
			index++
			if index == k {
				return current
			}
			continue
		}
		for i := 'c'; i >= 'a'; i-- {
			if len(current) == 0 || current[len(current)-1] != uint8(i) {
				stringStack = append(stringStack, current+string(i))
			}
		}
	}
	return ""
}

func maximumsSplicedArray(nums1 []int, nums2 []int) int {
	sum1 := 0
	sum2 := 0
	maxProfit := 0
	maxProfit2 := 0
	currentSum := 0
	currentSum2 := 0
	for i := 0; i < len(nums1); i++ {
		sum1 += nums1[i]
		sum2 += nums2[i]
		temp := nums2[i] - nums1[i]
		temp2 := nums1[i] - nums2[i]
		currentSum = int(math.Max(float64(currentSum)+float64(temp), float64(temp)))
		maxProfit = int(math.Max(float64(maxProfit), float64(currentSum)))
		currentSum2 = int(math.Max(float64(currentSum2)+float64(temp2), float64(temp2)))
		maxProfit2 = int(math.Max(float64(maxProfit2), float64(currentSum2)))
	}
	return int(math.Max(math.Max(float64(sum1+maxProfit), float64(sum1)), math.Max(float64(sum2+maxProfit2), float64(sum2))))
}

func poorPigs(buckets int, minutesToDie int, minutesToTest int) int {
	result := 0
	for math.Pow(float64((minutesToTest/minutesToDie)+1), float64(result)) < float64(buckets) {
		result++
	}
	return result
}

func checkAround(i, j int, matrix [][]int, visited [][]bool) int {
	if i >= 0 && i < len(matrix) && j >= 0 && j < len(matrix[0]) && !visited[i][j] && matrix[i][j] != 0 {
		visited[i][j] = true
	} else {
		return 0
	}
	return int(1 + checkAround(i+1, j, matrix, visited) + checkAround(i, j+1, matrix, visited) + checkAround(i-1, j, matrix, visited) + checkAround(i, j-1, matrix, visited))
}

func countServers(grid [][]int) int {
	var localCount int = 0
	var visited [][]bool = make([][]bool, len(grid))
	for index, _ := range grid {
		visited[index] = make([]bool, len(grid[index]))
	}
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			if grid[i][j] == 1 && !visited[i][j] {
				temp := int(checkAround(i, j, grid, visited))
				if temp != 1 {
					localCount += temp
				}
			}
		}
	}
	return localCount
}
func findDifferentBinaryString(nums []string) string {
	result := ""
	for i := 0; i < len(nums); i++ {
		if nums[i][i] == '0' {
			result += "1"
		} else {
			result += "0"
		}
	}
	return result
}

func sumOfTheDigitsOfHarshadNumber(x int) int {
	sumOfDigits := 0
	k := x
	for x > 0 {
		sumOfDigits += x % 10
		x /= 10
	}
	if k%sumOfDigits == 0 {
		return sumOfDigits
	}
	return -1
}

func numMovesStones(a int, b int, c int) []int {
	var minMove, maxMove int
	high := int(math.Max(float64(c), math.Max(float64(a), float64(b))))
	low := int(math.Min(float64(c), math.Min(float64(a), float64(b))))
	med := a + b + c - low - high
	if med-low > 1 {
		minMove++
	}
	if high-med > 1 {
		minMove++
	}
	maxMove = high - med - 1 + med - low - 1
	return []int{minMove, maxMove}
}

func modifyString(s string) string {
	str := strings.Split(s, "")
	chars := []string{"a", "b", "c"}

	for i := range str {
		if str[i] != "?" {
			continue
		}

		for _, c := range chars {
			if (i == 0 || str[i-1] != c) && (i == len(s)-1 || str[i+1] != c) {
				str[i] = c
				break
			}
		}
	}

	return strings.Join(str[:], "")
}

func mostWordsFound(sentences []string) int {
	maxSpaceCount := 0
	for i := 0; i < len(sentences); i++ {
		spaceCount := 0
		for j := 0; j < len(sentences[i]); j++ {
			if sentences[i][j] == ' ' {
				spaceCount++
			}
		}
		if spaceCount > maxSpaceCount {
			maxSpaceCount = spaceCount
		}
	}
	return maxSpaceCount + 1
}

/*
  - // This is the MountainArray's API interface.
  - // You should not implement it, or speculate about its implementation
  - type MountainArray struct {
  - }
    *
  - func (this *MountainArray) get(index int) int {}
  - func (this *MountainArray) length() int {}

import(
"fmt"
)

	func findInMountainArray(target int, mountainArr *MountainArray) int {
		n := mountainArr.length()
		if mountainArr.get(0) > target && mountainArr.get(n-1) > target{
			return -1
		}
		left := 0
		right := n - 1
		var peak int
		for left < right{
			mid := left + (right - left) / 2
			if mountainArr.get(mid) < mountainArr.get(mid + 1){
				left = mid + 1
			} else {
				right = mid
			}
		}
		fmt.Println(left)
		peak = left
		peakValue := mountainArr.get(peak)
		if target > peakValue{
			return  -1
		}

		if peakValue == target{
			return peak
		} else {
			result := -1
			if mountainArr.get(0) <= target {
				left = 0
				right = peak
				for left <= right{
					fmt.Println()
					fmt.Println(left)
					fmt.Println(right)
					med := left + (right-left)/2
					temp := mountainArr.get(med)
					if temp == target{
						result = med
						break
					} else if temp < target{
						left = med+1
					} else {
						right = med-1
					}
				}
				if result != -1 {
					return result
				}
			}
			if mountainArr.get(n-1) <= target {
				left = peak
				right = n-1
				for left <= right{
					med := left + (right-left)/2
					fmt.Println()
					fmt.Println(left)
					fmt.Println(right)
					temp := mountainArr.get(med)
					if temp == target{
						result = med
						break
					} else if temp > target{
						left = med+1
					} else {
						right = med-1
					}
				}
			}
			return result
		}
	}
*/

//// SeatManager seat construction
//type SeatManager struct {
//	seats []bool
//	min   int
//}
//
//func SeatConstructor(n int) SeatManager {
//	return SeatManager{make([]bool, n+1), 1}
//}
//
//func (this *SeatManager) Reserve() int {
//	for this.seats[this.min] {
//		this.min++
//	}
//	this.seats[this.min] = true
//	return this.min
//}
//
//func (this *SeatManager) Unreserve(seatNumber int) {
//	this.seats[seatNumber] = false
//	if seatNumber < this.min {
//		this.min = seatNumber
//	}
//}

func numberOfSubarrays(nums []int, k int) int {
	prefixOddCount := make([]int, len(nums)+1)
	temp := 0
	for i := 0; i < len(nums)+1; i++ {
		prefixOddCount[i] = temp
		if i < len(nums) && nums[i]%2 == 1 {
			temp++
		}
	}
	result := 0
	left := 0
	right := 1
	for right < len(prefixOddCount) {
		if prefixOddCount[right]-prefixOddCount[left] == k {
			i := left + 1
			j := right + 1
			for prefixOddCount[i] == prefixOddCount[left] && i < right {
				i++
			}
			for j < len(prefixOddCount) && prefixOddCount[j] == prefixOddCount[right] {
				j++
			}
			result += (i - left) * (j - right)
			left = i
			right = j

		} else if prefixOddCount[right]-prefixOddCount[left] < k {
			right++
		}
	}
	return result
}

func secondHighest(s string) int {
	largestTwo := []int{-1, -1}
	for i := 0; i < len(s); i++ {
		num, err := strconv.Atoi(string(s[i]))
		if err != nil {
			continue
		}
		if num > largestTwo[0] {
			largestTwo[1] = largestTwo[0]
			largestTwo[0] = num
		} else if num < largestTwo[0] {
			if num > largestTwo[1] {
				largestTwo[1] = num
			}
		}
	}
	return largestTwo[1]
}

func canArrange(arr []int, k int) bool {
	storage := make([]int, k)
	for i := 0; i < len(arr); i++ {
		storage[((arr[i]%k)+k)%k]++
	}
	if storage[0]%2 == 1 {
		return false
	}
	for i := 1; i < len(storage); i++ {
		if storage[i] != storage[k-i] {
			return false
		}
	}
	return true
}

func countSubarrays(nums []int, k int) int64 {
	maxEle := 0
	for i := range nums {
		maxEle = max(maxEle, nums[i])
	}
	var res int64 = 0
	count := 0
	left := 0
	for right := range nums {
		if nums[right] == maxEle {
			count++
		}
		for left <= right && count >= k {
			res += int64(len(nums) - right)
			if nums[left] == maxEle {
				count--
			}
			left++
		}
	}
	return res
}

func distMoney(money int, children int) int {
	switch {
	case children == 0 || money < children:
		return -1
	case children == 1 && money == 4:
		return -1
	case children == 1 && money == 8:
		return 1
	case children == 1:
		return 0
	}

	dpPrevious := distMoney(money-8, children-1)
	switch {
	case dpPrevious == -1 && children >= 2:
		return 0
	case dpPrevious == -1:
		return -1
	default:
		return 1 + dpPrevious
	}
}

func waysToBuyPensPencils(total int, cost1 int, cost2 int) int64 {
	result := int64(0)
	pencil := 0
	for total-pencil*cost1 >= 0 {
		result += int64((total-pencil*cost1)/cost2 + 1)
		pencil++
	}
	return result
}

func numOfSubarrays(arr []int) int {
	modulo := int64(math.Pow(10, 9) + 7)
	result := 0
	prefixSum := make([]int, len(arr)+1)
	prefixSum[0] = 0
	oddCount := 0
	evenCount := 1
	for i := 1; i < len(prefixSum); i++ {
		prefixSum[i] = prefixSum[i-1] + arr[i-1]
		if prefixSum[i]%2 == 0 {
			evenCount++
			result += oddCount
		} else {
			oddCount++
			result += evenCount
		}
	}

	return int(int64(result) % modulo)
}

func maxAbsoluteSum(nums []int) int {
	minPrefixSum := 0
	maxPrefixSum := 0
	prefixSum := 0
	for _, num := range nums {
		prefixSum += num
		minPrefixSum = min(minPrefixSum, prefixSum)
		maxPrefixSum = max(maxPrefixSum, prefixSum)
	}

	return maxPrefixSum - minPrefixSum
}

func search(nums []int, target int) int {
	left, right := 0, len(nums)-1
	for left <= right {
		mid := left + (right-left)/2
		if nums[mid] == target {
			return mid
		} else {
			if nums[mid] > target {
				right = mid - 1
			} else {
				left = mid + 1
			}
		}
	}
	return -1
}

func hIndex(citations []int) int {
	left, right := 0, len(citations)-1
	for left <= right {
		mid := left + (right-left)/2
		if citations[mid] >= len(citations)-mid {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}
	return len(citations) - left
}

func canCompleteCircuit(gas []int, cost []int) int {
	totalGas, totalCost, startingPoint, currentGas := 0, 0, 0, 0
	for i := 0; i < len(gas); i++ {
		totalGas += gas[i]
		totalCost += cost[i]
		currentGas += gas[i] - cost[i]
		if currentGas < 0 {
			startingPoint = i + 1
			currentGas = 0
		}
	}
	if totalGas < totalCost {
		return -1
	} else {
		return startingPoint
	}
}

func transformArray(nums []int) []int {
	result := make([]int, len(nums))
	indexFromLast := len(nums) - 1
	indexFromFirst := 0
	for i := 0; i < len(nums) && indexFromFirst <= indexFromLast; i++ {
		if nums[i]%2 == 1 {
			result[indexFromLast] = 1
			indexFromLast--
		} else {
			result[indexFromFirst] = 0
			indexFromFirst++
		}
	}
	return result
}
func countArrays(original []int, bounds [][]int) int {
	leftBound, rightBound := bounds[0][0], bounds[0][1]
	for i := 1; i < len(bounds); i++ {
		diff := original[i] - original[i-1]
		leftBound = leftBound + diff
		rightBound = rightBound + diff
		var r, z = mutualInterval(leftBound, rightBound, bounds[i][0], bounds[i][1])
		if r == 0 && z == 0 {
			return 0
		}
		leftBound = r
		rightBound = z
	}
	return rightBound - leftBound + 1
}
func mutualInterval(a, b, c, d int) (int, int) {
	// Check if there is an overlap
	if b >= c && d >= a {
		// Compute the mutual interval
		start := max(a, c)
		end := min(b, d)
		return start, end
	}

	// No overlap
	return 0, 0
}

// Helper function to compute the minimum of two integers
func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

// Helper function to compute the maximum of two integers
func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}

func largestInteger(nums []int, k int) int {
	if len(nums) == 1 {
		return nums[0]
	}
	globalCounter := make(map[int]int)
	localCounter := make(map[int]int)
	for i := 0; i < k; i++ {
		localCounter[nums[i]]++
	}
	for key, _ := range localCounter {
		globalCounter[key]++
	}
	result := -1
	for i := k; i < len(nums); i++ {
		localCounter[nums[i]]++
		localCounter[nums[i-k]]--
		if localCounter[nums[i-k]] == 0 {
			delete(localCounter, nums[i-k])
		}
		for key, _ := range localCounter {
			globalCounter[key]++
		}
	}
	for key, value := range globalCounter {
		if value == 1 && key > result {
			result = key
		}
	}
	return result
}

func mergeArrays(nums1 [][]int, nums2 [][]int) [][]int {
	m, n, firstPointer, lastPointer := len(nums1), len(nums2), 0, 0
	result := make([][]int, 0)
	for firstPointer < m || lastPointer < n {
		temp := make([]int, 2)
		var index int
		if firstPointer >= m {
			index = nums2[lastPointer][0]
		} else if lastPointer >= n {
			index = nums1[firstPointer][0]
		} else {
			index = min(nums1[firstPointer][0], nums2[lastPointer][0])
		}

		temp[0] = index
		if firstPointer < m && nums1[firstPointer][0] == index {
			temp[1] += nums1[firstPointer][1]
			firstPointer++
		}
		if lastPointer < n && nums2[lastPointer][0] == index {
			temp[1] += nums2[lastPointer][1]
			lastPointer++
		}
		result = append(result, temp)
	}
	return result
}

func pivotArray(nums []int, pivot int) []int {
	less, equal, bigger := make([]int, 0), make([]int, 0), make([]int, 0)
	for i := 0; i < len(nums); i++ {
		if nums[i] < pivot {
			less = append(less, nums[i])
		} else if nums[i] > pivot {
			bigger = append(bigger, nums[i])
		} else {
			equal = append(equal, nums[i])
		}
	}
	for _, value := range equal {
		less = append(less, value)
	}
	for _, value := range bigger {
		less = append(less, value)
	}
	return less
}

func validSquare(p1 []int, p2 []int, p3 []int, p4 []int) bool {
	var diff12, diff13, diff14, diff23, diff24, diff34 int64
	coll := [][]int{p1, p2, p3, p4}
	for i := 0; i < len(coll)-1; i++ {
		for j := i + 1; j < len(coll[i]); j++ {
			if coll[i][0] == coll[j][0] && coll[i][1] == coll[j][1] {
				return false
			}
		}
	}
	dict := make(map[int64]int)
	diff12 = int64(math.Abs(float64(p1[0]-p2[0]))*math.Abs(float64(p1[0]-p2[0])) + math.Abs(float64(p1[1]-p2[1]))*math.Abs(float64(p1[1]-p2[1])))
	dict[diff12]++

	diff14 = int64(math.Abs(float64(p1[0]-p4[0]))*math.Abs(float64(p1[0]-p4[0])) + math.Abs(float64(p1[1]-p4[1]))*math.Abs(float64(p1[1]-p4[1])))
	dict[diff14]++

	diff23 = int64(math.Abs(float64(p2[0]-p3[0]))*math.Abs(float64(p2[0]-p3[0])) + math.Abs(float64(p2[1]-p3[1]))*math.Abs(float64(p2[1]-p3[1])))
	dict[diff23]++

	diff34 = int64(math.Abs(float64(p4[0]-p3[0]))*math.Abs(float64(p4[0]-p3[0])) + math.Abs(float64(p4[1]-p3[1]))*math.Abs(float64(p4[1]-p3[1])))
	dict[diff34]++

	diff13 = int64(math.Abs(float64(p1[0]-p3[0]))*math.Abs(float64(p1[0]-p3[0])) + math.Abs(float64(p1[1]-p3[1]))*math.Abs(float64(p1[1]-p3[1])))
	dict[diff13]++

	diff24 = int64(math.Abs(float64(p2[0]-p4[0]))*math.Abs(float64(p2[0]-p4[0])) + math.Abs(float64(p2[1]-p4[1]))*math.Abs(float64(p2[1]-p4[1])))
	dict[diff24]++
	if len(dict) != 2 {
		return false
	} else {
		var equalLength int64 = math.MaxInt64
		var maxLength int64 = math.MinInt64
		for key, _ := range dict {
			if key < equalLength {
				equalLength = key
			}
			if key > maxLength {
				maxLength = key
			}
		}
		if dict[equalLength] == 4 && dict[maxLength] == 2 {
			return true
		} else {
			return false
		}
	}
}

func coloredCells(n int) int64 {
	return int64(2*n*n - 2*n + 1)
}

func findMissingAndRepeatedValues(grid [][]int) []int {
	n := len(grid)
	result := make([]int, 2)
	sum := 0
	squareSum := int64(0)
	idealSum := (n * n) * (n*n + 1) / 2
	idealSquareSum := int64((n*n)*(n*n+1)*(2*n*n+1)) / 6
	for i := 0; i < len(grid); i++ {
		for j := 0; j < len(grid[i]); j++ {
			sum += grid[i][j]
			squareSum += int64(grid[i][j] * grid[i][j])
		}
	}
	diff := int64(sum - idealSum)
	diffSquareSum := squareSum - idealSquareSum
	result[0] = int(diffSquareSum/diff+diff) / 2
	result[1] = result[0] - int(diff)
	return result
}
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func minOperations(boxes string) []int {
	result := make([]int, len(boxes))
	for i := 0; i < len(boxes); i++ {
		for j := 0; j < len(boxes); j++ {
			if j != i && boxes[j] == '1' {
				result[i] += abs(j - i)
			}
		}
	}
	return result
}

func scoreOfString(s string) int {
	sum := 0
	for i := 1; i < len(s); i++ {
		sum += abs(int(s[i]) - int(s[i-1]))
	}
	return sum
}

func findArray(pref []int) []int {
	result := make([]int, len(pref))
	result[0] = pref[0]
	for i := 1; i < len(pref); i++ {
		result[i] = result[i-1] ^ pref[i]
	}
	return result
}

//func maximalRectangle(matrix [][]byte) int {
//	arr := matrix[0]
//	for i := 0; i < len(matrix); i++ {
//
//	}
//}

func largestRectangleArea(heights []int) int {
	var stack []int
	maxArea := 0
	n := len(heights)

	for i := 0; i <= n; i++ {
		currentHeight := 0
		if i < n {
			currentHeight = heights[i]
		}

		for len(stack) > 0 && heights[stack[len(stack)-1]] > currentHeight {
			height := heights[stack[len(stack)-1]]
			stack = stack[:len(stack)-1]
			width := i
			if len(stack) > 0 {
				width = i - stack[len(stack)-1] - 1
			}
			maxArea = max(maxArea, height*width)
		}

		stack = append(stack, i)
	}

	return maxArea
}

func maxScoreSightseeingPair(values []int) int {
	maxScore := 0
	bestLeft := values[0]
	for j := 1; j < len(values); j++ {
		maxScore = max(maxScore, bestLeft+values[j]-j)
		bestLeft = max(bestLeft, values[j]+j)
	}
	return maxScore
}

func climbStairs(n int) int {
	dp := make([]int, n+1)
	dp[0] = 1
	dp[1] = 1
	for i := 2; i <= n; i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}
func getDescentPeriods(prices []int) int64 {
	var result int64 = 1
	dp := make([]int, len(prices))
	dp[0] = 1
	for i := 1; i < len(prices); i++ {
		if prices[i] == prices[i-1]-1 {
			dp[i] = dp[i-1] + 1
			result += int64(dp[i])
		} else {
			dp[i] = 1
			result += int64(dp[i])
		}
	}
	return result
}

func sieveOfEratosthenes(n1, n2 int) []int {
	prime := make([]bool, n2+1)
	for i := range prime {
		prime[i] = true
	}
	p := 2
	for p*p <= n2 {
		if prime[p] {
			for i := p * p; i <= n2; i += p {
				prime[i] = false
			}
		}
		p++
	}
	var primes []int
	for p := n1; p <= n2; p++ {
		if prime[p] && p != 1 {
			primes = append(primes, p)
		}
	}

	return primes
}

func closestPrimes(left int, right int) []int {
	primes := sieveOfEratosthenes(left, right)
	if len(primes) < 2 {
		return []int{-1, -1}
	} else {
		result := []int{primes[0], primes[1]}
		for i := 2; i < len(primes); i++ {
			if primes[i]-primes[i-1] < result[1]-result[0] {
				result[0] = primes[i-1]
				result[1] = primes[i]
			}
		}
		return result
	}
}

func divideArray(nums []int) bool {
	var mapper = make(map[int]int)
	for i := 0; i < len(nums); i++ {
		mapper[nums[i]]++
	}
	for _, value := range mapper {
		if value%2 == 1 {
			return false
		}
	}
	return true
}

func longestNiceSubarray(nums []int) int {
	n := len(nums)
	maxLength := 1
	left := 0
	usedBits := 0

	for right := 0; right < n; right++ {
		for (usedBits & nums[right]) != 0 {
			usedBits ^= nums[left]
			left++
		}

		usedBits |= nums[right]
		if right-left+1 > maxLength {
			maxLength = right - left + 1
		}
	}

	return maxLength
}

func countSymmetricIntegers(low int, high int) int {
	result := 0
	for num := low; num <= high; num++ {
		if num < 100 {
			if num%11 == 0 {
				result++
			}
		} else if num >= 1000 && num < 10000 {
			left := num/1000 + (num%1000)/100
			right := (num%100)/10 + num%10
			if left == right {
				result++
			}
		}
	}
	return result
}

func isArraySpecial(nums []int) bool {

	for i := 0; i < len(nums)-1; i++ {
		if (nums[i]+nums[i+1])%2 == 0 {
			return false
		}
	}
	return true
}

func generate(numRows int) [][]int {
	result := make([][]int, numRows)
	for index := range result {
		result[index] = make([]int, index+1)
		for i := 0; i < len(result[index]); i++ {
			if i == 0 || i == index {
				result[index][i] = 1
			} else {
				result[index][i] = result[index-1][i] + result[index-1][i-1]
			}
		}
	}
	return result
}

var triangles [][]int = buildTriangles()

func buildTriangles() [][]int {
	result := make([][]int, 33)
	for index := range result {
		result[index] = make([]int, index+1)
		for i := 0; i < len(result[index]); i++ {
			if i == 0 || i == index {
				result[index][i] = 1
			} else {
				result[index][i] = result[index-1][i] + result[index-1][i-1]
			}
		}
	}
	return result
}

func maxProfit(prices []int) int {
	result, buyDay := 0, prices[0]
	for i := 1; i < len(prices); i++ {
		if prices[i]-buyDay > result {
			result = prices[i] - buyDay
		}
		if prices[i] < buyDay {
			buyDay = prices[i]
		}
	}
	return result
}

func maxProfit2(prices []int) int {
	result := 0
	for i := 1; i < len(prices); i++ {
		if prices[i]-prices[i-1] > 0 {
			result += prices[i] - prices[i-1]
		}
	}
	return result
}

func getRow(rowIndex int) []int {
	return triangles[rowIndex]
}

func searchMatrix(matrix [][]int, target int) bool {
	m, n := len(matrix), len(matrix[0])

	left, right := 0, m-1
	for left <= right {
		mid := left + (right-left)>>1
		if matrix[mid][n-1] >= target && matrix[mid][0] <= target {
			innerLeft, innerRight := 0, n-1
			for innerLeft <= innerRight {
				innerMid := innerLeft + (innerRight-innerLeft)>>1
				if matrix[mid][innerMid] == target {
					return true
				} else if matrix[mid][innerMid] > target {
					innerRight = innerMid - 1
				} else {
					innerLeft = innerMid + 1
				}
			}
			return false
		} else if matrix[mid][0] > target {
			right = mid - 1
		} else if matrix[mid][n-1] < target {
			left = mid + 1
		}
	}

	return false
}

func getCombination(size int) int {
	return size * (size - 1) / 2
}

func countBits(n int) []int {
	result := make([]int, n+1)
	result[0] = 0
	sub := 1
	for i := 1; i <= n; i++ {
		if sub*2 == i {
			sub = i
		}

		result[i] = result[i-sub] + 1
	}
	return result
}

func countVowelStrings(n int) int {
	dp := make([][]int, n+1)
	dp[0] = []int{0, 0, 0, 0, 0}
	dp[1] = []int{1, 1, 1, 1, 1}
	sum := 5
	for i := 2; i <= n; i++ {
		dp[i] = make([]int, 5)
		for j := 0; j < 5; j++ {
			if j == 0 {
				dp[i][j] = sum
				sum = 0
			} else {
				dp[i][j] = dp[i][j-1] - dp[i-1][j-1]
			}
			sum += dp[i][j]
		}
	}

	return sum
}

func mincostTickets(days []int, costs []int) int {
	lastDay := days[len(days)-1]
	dp := make([]int, lastDay+1)
	dp[0] = 0
	i := 0
	for day := 1; i <= lastDay; i++ {
		if day < days[0] {
			dp[day] = dp[day-1]
		} else {
			dp[day] = min(dp[day-1]+costs[0],
				min(dp[max(day-7, 0)]+costs[1], dp[max(day-30, 0)]+costs[2]))
		}
	}
	return dp[lastDay]
}

func numJewelsInStones(jewels string, stones string) int {
	jewelMap := make(map[byte]int)
	result := 0
	for i := 0; i < len(jewels); i++ {
		jewelMap[jewels[i]] = jewelMap[jewels[i]] + 1
	}
	for i := 0; i < len(stones); i++ {
		if jewelMap[stones[i]] >= 1 {
			result++
		}
	}
	return result
}

func minEatingSpeed(piles []int, h int) int {
	canKokoEatAt := func(speed int) bool {
		sum := 0
		for i := 0; i < len(piles); i++ {
			sum += (piles[i]-1)/speed + 1
		}
		return sum <= h
	}

	left, right := 0, slices.Max(piles)
	for left < right {
		mid := left + (right-left)>>1
		if canKokoEatAt(mid) {
			right = mid
		} else {
			left = mid + 1
		}
	}

	return left
}

func smallestNumber(n int, t int) int {

	getDigitProduct := func(n int) int {
		sum := 1
		for n > 0 {
			sum *= n % 10
			n /= 10
		}
		return sum
	}

	for i := n; i <= n+9; i++ {
		if getDigitProduct(i)%t == 0 {
			return i
		}
	}

	return n
}

func nextGreatestLetter(letters []byte, target byte) byte {
	left, right := 0, len(letters)-1
	for left < right {
		mid := left + (right-left)>>1
		if letters[mid] == target {
			if letters[mid+1] != target {
				return letters[mid+1]
			} else {
				left = mid + 1
			}
		} else if letters[mid] < target {
			if letters[mid+1] > target {
				return letters[mid+1]
			}
			left = mid + 1
		} else {
			right = mid
		}

	}

	return letters[0]
}

func countSubarrays2302(nums []int, k int64) int64 {
	n := len(nums)
	left, right := 0, 0
	sum := int64(0)
	res := int64(0)
	for ; right < n; right++ {
		sum += int64(nums[right])
		for left <= right && sum*int64(right-left+1) >= k {
			sum -= int64(nums[left])
			left++
		}

		res += int64(right - left + 1)
	}

	return res
}

func numRabbits(answers []int) int {
	sum := 0
	rabbitCount := make(map[int]int)
	for i := 0; i < len(answers); i++ {
		rabbitCount[answers[i]]++
	}

	for k, v := range rabbitCount {
		sum += int(math.Ceil(float64(v)/float64(k+1))) * (k + 1)
	}

	return sum
}

func maximumSubarraySum(nums []int, k int) int64 {
	maxSum := int64(0)

	for left := 0; left <= len(nums)-k; {
		sum := int64(0)
		lengthK := false
		mapper := make(map[int]int)
		for ptr := left; ptr < left+k; ptr++ {
			mapper[nums[ptr]]++
			sum += int64(nums[ptr])
			if mapper[nums[ptr]] > 1 {
				for i := 0; i <= ptr; i++ {
					if nums[i] == nums[ptr] {
						left = ptr - 1
						break
					}
				}
				break
			}
			if ptr == left+k-1 {
				lengthK = true
			}
		}
		if sum > maxSum && lengthK {
			maxSum = sum
		}
	}
	return maxSum
}

func maxCount(banned []int, n int, maxSum int) int {
	sum := 0
	i := 1
	count := 0
	ban := make([]bool, n+1)
	for _, v := range banned {
		if v > n {
			continue
		}
		ban[v] = true
	}

	for ; i <= n; i++ {
		if ban[i] {
			continue
		}
		sum += i
		if sum > maxSum {
			break
		}
		count++
	}

	return count
}

func alphabetBoardPath(target string) string {
	r, c := 0, 0
	s := ""
	for _, v := range target {
		row := int((v - 'a') / 5)
		col := int((v - 'a') % 5)
		for r < row {
			if r == 4 {
				for c > 0 {
					s += "L"
					c--
				}
			}
			s += "D"
			r++

		}
		for r > row {
			s += "U"
			r--
		}
		for c < col {
			if r == 5 {
				s += "U"
				r--
			} else {
				s += "R"
				c++
			}
		}
		for c > col {
			s += "L"
			c--
		}
		s += "!"
	}
	return s
}

func lengthOfLongestSubstring(s string) int {
	if len(s) == 0 {
		return 0
	}
	result := 0
	mapper := make(map[rune]int)
	left, right := 0, 0
	for ; right < len(s); right++ {
		mapper[rune(s[right])]++
		if mapper[rune(s[right])] > 1 {
			result = max(result, right-left)
			for left < right {
				if mapper[rune(s[left])] == mapper[rune(s[right])] {
					mapper[rune(s[left])]--
					left++
					break
				}
				mapper[rune(s[left])]--

				left++
			}
		}
	}
	return max(result, right-left)
}

func isZeroArray(nums []int, queries [][]int) bool {
	n := len(nums)
	del := make([]int, n+1)

	for _, q := range queries {
		l := q[0]
		r := q[1]
		del[l]++
		del[r+1]--
	}

	for i := 1; i <= n; i++ {
		del[i] += del[i-1]
	}

	for i := 0; i < n; i++ {
		if del[i] < nums[i] {
			return false
		}
	}
	return true
}

func countCompleteSubarrays(nums []int) int {
	set := map[int]bool{}
	for _, v := range nums {
		set[v] = true
	}
	total := len(set)
	count := 0
	freq := map[int]int{}
	l := 0

	for r, val := range nums {
		freq[val]++
		for len(freq) == total {
			count += len(nums) - r
			freq[nums[l]]--
			if freq[nums[l]] == 0 {
				delete(freq, nums[l])
			}
			l++
		}
	}
	return count
}

func findWordsContaining(words []string, x byte) []int {
	result := make([]int, 0)
	for i := 0; i < len(words); i++ {
		if strings.Contains(words[i], string(x)) {
			result = append(result, i)
		}
	}
	return result
}

func isValid(word string) bool {
	n := len(word)
	if n < 3 {
		return false
	} else {
		vowels := []rune{'a', 'e', 'i', 'o', 'u'}
		consonants := []rune{
			'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm',
			'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z',
		}
		vowel := false
		consonant := false
		for i := 0; i < n; i++ {
			if unicode.IsLetter(rune(word[i])) || unicode.IsNumber(rune(word[i])) {
				if slices.Contains(vowels, unicode.ToLower(rune(word[i]))) {
					vowel = true
				} else if slices.Contains(consonants, unicode.ToLower(rune(word[i]))) {
					consonant = true
				}
			} else {
				return false
			}

		}

		return vowel && consonant
	}

}

func main() {
	fmt.Println()
}

/* Randomized Set
type RandomizedSet struct {
	arr  []int
	size int
	set  map[int]int
}

func Constructor() RandomizedSet {
	arr := make([]int, 0)
	size := 0
	set := make(map[int]int)
	return RandomizedSet{
		arr:  arr,
		size: size,
		set:  set,
	}
}

func (this *RandomizedSet) Insert(val int) bool {
	_, ok := this.set[val]
	if ok {
		return false
	}

	this.arr = append(this.arr, val)
	this.set[val] = this.size
	this.size++
	return true
}

func (this *RandomizedSet) Remove(val int) bool {
	index, ok := this.set[val]
	if !ok {
		return false
	}

	// Swap the element to remove with the last element
	lastElement := this.arr[this.size-1]
	this.arr[index] = lastElement
	this.set[lastElement] = index

	// Remove the last element
	this.arr = this.arr[:this.size-1]
	delete(this.set, val)
	this.size--
	return true
}

func (this *RandomizedSet) GetRandom() int {
	index := rand.Intn(this.size)
	return this.arr[index]
}

*/
